#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/out/train_melgan_deeptestnet-256_%j.out
#SBATCH --error=/home/s6kogase/code/out/train_melgan_deeptestnet-256_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310

python -m src.new_train_classifier \
    --batch-size 256 \
    --learning-rate 0.001 \
    --weight-decay 0.000001   \
    --epochs 25 \
    --validation-interval 50    \
    --data-prefix "${HOME}/data/binary_cmor4.6-0.87_8000_8736_4368_224_2000-4000_1_10000_melgan" \
    --nclasses 2 \
    --seed 0 \
    --tensorboard \
    --model "deeptestnet"  \
    --num-workers 5

echo "Goodbye at $(date)."