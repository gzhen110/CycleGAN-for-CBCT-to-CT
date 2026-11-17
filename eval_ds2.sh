#!/bin/bash
#SBATCH --job-name=eval_ds2
#SBATCH --account=MST115280
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_ds2.txt

module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate medical

cd ~/CycleGAN-for-CBCT-to-CT
python evaluate_tf.py \
    --model_dir ./models/dataset2/ \
    --data /work/jhz21803/dataset2_npy/ \
    --out ./results/dataset2/ \
    --save_npy
