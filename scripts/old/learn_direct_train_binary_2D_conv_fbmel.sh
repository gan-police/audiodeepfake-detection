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
    --batch-size 128  \
    --learning-rate 0.001 \
    --weight-decay 0.000001   \
    --epochs 10 \
    --validation-interval 100    \
    --data-prefix "${HOME}/data/fake_cmor4.6-0.87_22050_8000_11025_224_80-4000_1_0.7_fbmelgan" \
    --nclasses 2 \
    --seed 0 \
    --tensorboard \
    --model "learndeepnet"  \
    --wavelet "cmor17-0.87" \
    --f-min 2000 \
    --f-max 10000 \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --num-workers 3

echo "Goodbye at $(date)."