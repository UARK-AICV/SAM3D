#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_tumor
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task03_tumor
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python3 ../sam3d/run/run_training.py 3d_fullres sam3d_trainer_tumor 3 0 --vit_name vit_b 
