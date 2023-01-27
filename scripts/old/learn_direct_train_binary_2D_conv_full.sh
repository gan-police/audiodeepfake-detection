#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/out/train_learnable_%j.out
#SBATCH --error=/home/s6kogase/code/out/train_learnable_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.learn_direct_train_classifier \
    --batch-size 200 \
    --learning-rate 0.001 \
    --weight-decay 0.000001   \
    --epochs 15 \
    --validation-interval 50    \
    --data-prefix "${HOME}/data/fake_cmor4.6-0.87_8000_8000_11025_224_2000-4000_1_0.7_melgan" \
    --nclasses 2 \
    --seed 0 \
    --tensorboard \
    --model "learndeepnet"  \
    --wavelet "cmor4.6-0.87" \
    --adapt-wavelet \
    --num-workers 0

echo "Goodbye at $(date)."