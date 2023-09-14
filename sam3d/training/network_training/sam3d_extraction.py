from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from sam3d.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from sam3d.training.loss_functions.deep_supervision import MultipleOutputLoss2
from sam3d.utilities.to_torch import maybe_to_torch, to_cuda
from sam3d.network_architecture.initialization import InitWeights_He
from sam3d.network_architecture.neural_network import SegmentationNetwork
from sam3d.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from sam3d.training.dataloading.dataset_loading import unpack_dataset
from sam3d.training.network_training.Trainer_acdc import Trainer_acdc
from sam3d.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from sam3d.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from sam3d.network_architecture.sam3d import Sam3D

class sam3d_trainer_acdc_extraction(Trainer_acdc):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, vit_name='vit_b', sam_ckpt=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False

        self.load_plans_file()

        if len(self.plans['plans_per_stage']) == 2:
            Stage = 1
        else:
            Stage = 0

        self.crop_size = self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1
        self.conv_op = nn.Conv3d

        self.deep_supervision = True
        self.vit_name = vit_name
        self.sam_ckpt = sam_ckpt
        self.threeD = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['plans_per_stage'][0]['patch_size'] = np.array([16, 160, 160])
            self.crop_size = np.array([16, 160, 160])

            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                # weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory,
                                                      self.plans['data_identifier'] + "_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        
        self.network = Sam3D(num_classes=self.num_classes, ckpt=self.sam_ckpt, image_size=self.crop_size, vit_name=self.vit_name, conv_op=self.conv_op, num_modalities=self.input_channels, do_ds=True)
            
        # Print the network parameters
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        # print(data.shape)
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # import pdb; pdb.set_trace()
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data

                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()