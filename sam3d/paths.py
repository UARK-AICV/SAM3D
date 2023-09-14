#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "sam3d"
default_plans_identifier = "sam3d_Plansv2.1"
default_data_identifier = 'sam3d_Data_plans_v2.1'
default_trainer = "sam3d_trainer_acdc"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base = os.environ['sam3d_raw_data_base'] if "sam3d_raw_data_base" in os.environ.keys() else None
preprocessing_output_dir = os.environ['sam3d_preprocessed'] if "sam3d_preprocessed" in os.environ.keys() else None
network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None

if base is not None:
    nnFormer_raw_data = join(base, "sam3d_raw_data")
    nnFormer_cropped_data = join(base, "sam3d_cropped_data")
    maybe_mkdir_p(nnFormer_raw_data)
    maybe_mkdir_p(nnFormer_cropped_data)
else:
    print("sam3d_raw_data_base is not defined and model can only be used on data for which preprocessed files "
          "are already present on your system. model cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read run_training_synapse.sh/run_training_acdc.sh "
          "for information on how to set this up properly.")
    nnFormer_cropped_data = nnFormer_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("sam3d_preprocessed is not defined and model can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for "
          "information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read run_training_synapse.sh/run_training_acdc.sh "
          "for information on how to set this up.")
    network_training_output_dir = None
