#!/bin/sh
#SBATCH --job-name=test_acdc         # create a short name for your job
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

DATASET_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Acdc
CHECKPOINT_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Acdc/output_acdc

export PYTHONPATH=/home/hdhieu/3DSAM-Decoder-1
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task01_ACDC
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

# python3 /home/bntan/3DSAM-Decoder-1/sam3d/inference/predict_simple.py -i /home/bntan/3DSAM-Decoder-1/DATASET_Acdc/sam3d_raw/sam3d_raw_data/Task001_ACDC/imagesTs -o /home/bntan/3DSAM-Decoder-1/output/ACDC -t 1 -m 3d_fullres -tr sam3d_trainer_acdc 
# python3  /home/hdhieu/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_acdc 1 0 -val --val_folder testing_best_post --valbest
# python3 /home/bntan/3DSAM-Decoder-1/sam3d/inference_acdc.py 
