#!/bin/sh
#SBATCH --job-name=train_brain         # create a short name for your job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=selab2
#SBATCH --mem-per-cpu=4GB  

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK


DATASET_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Tumor

export PYTHONPATH=/home/hdhieu/3DSAM-Decoder-1
export RESULTS_FOLDER=/home/hdhieu/3DSAM-Decoder-1/DATASET_Tumor/output_tumor
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task03_tumor
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python3 /home/hdhieu/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_tumor 3 0 -c
