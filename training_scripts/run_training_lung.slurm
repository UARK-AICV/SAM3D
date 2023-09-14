#!/bin/sh
#SBATCH --job-name=train_lung        # create a short name for your job
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

DATASET_PATH=/home/bntan/3DSAM-Decoder/DATASET_Lungs

export PYTHONPATH=/home/bntan/3DSAM-Decoder-1
export RESULTS_FOLDER=/home/bntan/3DSAM-Decoder-1/DATASET_Lung/output_lung
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task06_Lung
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python3 /home/bntan/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_lung 6 0 -c
