#!/bin/sh
#SBATCH --job-name=train_synapse         # create a short name for your job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10-00:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=phoenix2

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK

export PATH="/home/hdhieu/miniconda3/bin:$PATH"
# conda init bash

source activate 3DSAM

DATASET_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Synapse

export PYTHONPATH=/home/hdhieu/3DSAM-Decoder-1
export RESULTS_FOLDER=/home/hdhieu/3DSAM-Decoder-1/DATASET_Synapse/output_synapse
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task02_Synapse
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python3 /home/hdhieu/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_synapse 2 3
