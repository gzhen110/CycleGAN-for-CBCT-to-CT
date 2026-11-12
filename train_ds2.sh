#!/bin/bash
#SBATCH --job-name=train_ds2
#SBATCH --account=MST115280
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=47:30:00
#SBATCH --output=logs/train_ds2_loss_%j.txt
#SBATCH --error=logs/train_ds2_progress_%j.txt

# 同一支腳本可以重複送出：models/dataset2/ 裡有 last.ckpt 時會自動接續訓練
# %j 會換成工作編號，每一段訓練各自一份 log，不會互相覆蓋
# （完整的 loss 紀錄另外存在 models/dataset2/loss_log.csv，會跨段接續）

module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate medical

cd ~/CycleGAN-for-CBCT-to-CT
python main.py \
    -d /work/jhz21803/dataset2_npy/ \
    -m ./models/dataset2/ \
    --height 304 --width 304 \
    -b 16 \
    --epoch 200 \
    --pool_size 50 \
    --workers 8
