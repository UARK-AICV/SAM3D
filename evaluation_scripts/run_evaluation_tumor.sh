#!/bin/sh

DATASET_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Tumor
CHECKPOINT_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Tumor/output_tumor

export PYTHONPATH=/home/hdhieu/3DSAM-Decoder-1
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task03_tumor
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

# Only for Tumor, it is recommended to train unetr_plus_plus first, and then use the provided checkpoint to evaluate. It might raise issues regarding the pickle files if you evaluated without training

python3 /home/hdhieu/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_tumor 3 0 -val --valbest --val_folder testing_best

