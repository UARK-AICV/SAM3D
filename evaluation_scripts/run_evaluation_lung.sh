#!/bin/sh
#SBATCH --job-name=test_lung         # create a short name for your job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=selab2
#SBATCH --mem-per-cpu=4GB   

DATASET_PATH=/home/hdhieu/3DSAM-Decoder-Ablation/DATASET_Lungs
CHECKPOINT_PATH=/home/hdhieu/3DSAM-Decoder-1/DATASET_Lung/output_lung

export PYTHONPATH=/home/hdhieu/3DSAM-Decoder-1
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export sam3d_preprocessed="$DATASET_PATH"/sam3d_raw/sam3d_raw_data/Task06_Lung
export sam3d_raw_data_base="$DATASET_PATH"/sam3d_raw

python3  /home/hdhieu/3DSAM-Decoder-1/sam3d/run/run_training.py 3d_fullres sam3d_trainer_lung 6 0 -val --val_folder testing_final
