#!/bin/sh

DATASET_PATH=.../DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=.../output_acdc
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task01_ACDC
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python .../sam3d/run/run_training.py 3d_fullres sam3d_trainer_acdc 1 0