#!/bin/bash
#SBATCH --job-name=checkbg_ds2
#SBATCH --account=MST115280
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/check_bg_ds2.txt

module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate medical

cd ~/CycleGAN-for-CBCT-to-CT
python check_background.py \
    --model_dir ./models/dataset2/ \
    --ckpt epoch_200.ckpt \
    --data /work/jhz21803/dataset2_npy/ \
    --out ./results/dataset2/check_bg/
