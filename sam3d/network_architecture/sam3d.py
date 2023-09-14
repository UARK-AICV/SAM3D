from sam3d.network_architecture.segment_anything import build_sam, SamPredictor
from sam3d.network_architecture.segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from sam3d.network_architecture.segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from sam3d.network_architecture.neural_network import SegmentationNetwork
from sam3d.network_architecture.segment_anything import sam_model_registry
from einops import repeat, rearrange
import time 


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kSize=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True, mode='nearest'):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels), 
            nn.LeakyReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )
        
        self.leakyrelu = nn.LeakyReLU()
        
        self.up = nn.Upsample(scale_factor=(1,2,2), mode=mode)
        
    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        
        out = self.leakyrelu(out)
        out = self.up(out)
        return out

class BasicBlockTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kSize=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True, mode='nearest'):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels), 
            nn.LeakyReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )
        
        self.residual = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )
        
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        
        out = self.leakyrelu(out)
        return out
    
    
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kSize=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True):
        super().__init__()
        
        self.conv_pred = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(in_channels // 2), 
            nn.LeakyReLU()
        )
        self.segmentation_head = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv_pred(x)
        return self.segmentation_head(x)
        

class Sam3D(SegmentationNetwork):
    def __init__(
        self,
        num_classes,
        ckpt,
        image_size,
        vit_name='vit_b',
        conv_op=nn.Conv3d,
        num_modalities=1,
        do_ds=True,
    ) -> None:
        """
        Args:
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """
        super().__init__()
        
        self.vit_name = vit_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.sam_ckpt = ckpt
        self.do_ds = do_ds
        self.num_modalities = num_modalities
        
        if self.num_modalities > 1:
            self.decoder5 = BasicBlock(in_channels=256 * self.num_modalities, out_channels=128)
        else:
            self.decoder5 = BasicBlock(in_channels=256, out_channels=128)
        
        self.decoder4 = BasicBlock(in_channels=128, out_channels=64)
        
        self.decoder3 = BasicBlock(in_channels=64, out_channels=32)
        
        self.decoder2 = BasicBlock(in_channels=32, out_channels=16)
        
        self.out1 = SegmentationHead(in_channels=16, out_channels=self.num_classes)
        if do_ds:
            self.out3 = SegmentationHead(in_channels=64, out_channels=self.num_classes)
            self.out2 = SegmentationHead(in_channels=32, out_channels=self.num_classes)

    def forward(self, x):
        B, C, D, H, W = x.shape  
        sam_size = self.image_size[1]
        
        sam = sam_model_registry[self.vit_name](image_size=sam_size,
                                                num_classes=self.num_classes,
                                                checkpoint=self.sam_ckpt, in_channel=3,
                                                pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]).cuda()
        
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        
        list_feature_encoder = []
        
        if C > 1:
            list_feature_channel = []
            for channel in range(C):
                for dep in range(D):
                    with torch.no_grad():
                        image = x[:,channel,dep,:,:]
                        image = rearrange(image, '(b c) h w -> b c h w', c=1)
                        image = repeat(image, 'b c h w -> b (repeat c) h w', repeat=3)
                        feature = sam.image_encoder(image)
                        list_feature_encoder.append(feature)
                
                all_feature_channel = torch.stack(list_feature_encoder, 2)
                list_feature_encoder = []
                list_feature_channel.append(all_feature_channel)
                
            all_feature_encoder = torch.stack(list_feature_channel, 1)
            all_feature_encoder = rearrange(all_feature_encoder, 'b m c d h w -> b (m c) d h w')    
        else:
            for dep in range(D):
                with torch.no_grad():
                    image = x[:,:,dep,:,:]
                    image = repeat(image, 'b c h w -> b (repeat c) h w', repeat=3)
                    feature = sam.image_encoder(image)
                    list_feature_encoder.append(feature)
        
            all_feature_encoder = torch.stack(list_feature_encoder, 2)
        
        dec5 = self.decoder5(all_feature_encoder)
        dec4 = self.decoder4(dec5)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)

        if self.do_ds:
            logits = [self.out1(dec2), self.out2(dec3), self.out3(dec4)]
        else:
            logits = self.out1(dec2)
            
        return logits