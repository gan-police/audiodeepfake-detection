#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/out/train_melgan_onednet_512-%A_%a.out
#SBATCH --error=/home/s6kogase/code/out/train_melgan_onednet_512-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --partition=A40short
#SBATCH --array=0-6

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310

python -m src.new_train_classifier \
    --batch-size 512 \
    --learning-rate 0.001 \
    --weight-decay 0.000001   \
    --epochs 20 \
    --validation-interval 50    \
    --data-prefix "${HOME}/data/binary_cmor4.6-0.87_8000_8736_4368_224_2000-4000_1_10000_melgan" \
    --nclasses 2 \
    --seed $SLURM_ARRAY_TASK_ID \
    --model "onednet"  \
    --num-of-scales 224 \
    --tensorboard \
    --num-workers 5

echo "Goodbye at $(date)."