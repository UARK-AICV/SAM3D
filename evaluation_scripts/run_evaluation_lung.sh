#!/bin/sh

DATASET_PATH=../DATASET_Lungs
CHECKPOINT_PATH=../DATASET_Lung/output_lung

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task06_Lung
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python ../sam3d/run/run_training.py 3d_fullres sam3d_trainer_lung 6 0 -val --valbest
