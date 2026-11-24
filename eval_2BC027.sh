#!/bin/bash
#SBATCH --job-name=eval_2BC027
#SBATCH --account=MST115280
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_2BC027.txt

module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate medical

cd ~/CycleGAN-for-CBCT-to-CT
python evaluate_tf.py \
    --model_dir ./models/dataset2/ \
    --data /work/jhz21803/dataset2_npy/ \
    --out ./results/dataset2_2BC027/ \
    --patients 2BC027 \
    --save_npy