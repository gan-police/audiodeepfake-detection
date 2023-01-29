#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/out/train_learnable_onednet_%A_%a_0.out
#SBATCH --error=/home/s6kogase/code/out/train_learnable_onednet_%A_%a_0.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:00:00
#SBATCH --partition=A40short
#SBATCH --array=0-7

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
gans=("melgan" "hifigan"  "mbmelgan"  "fbmelgan"  "waveglow" "pwg" "lmelgan" "all")

conda activate py310
python -m src.learn_direct_train_classifier \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001   \
    --epochs 10 \
    --validation-interval 300    \
    --data-prefix "${HOME}/data/fake_cmor4.6-0.87_22050_8000_11025_224_80-4000_1_0.7_${gans[$SLURM_ARRAY_TASK_ID]}" \
    --nclasses 2 \
    --seed 0 \
    --tensorboard \
    --model "onednet"  \
    --wavelet "cmor3.3-4.17" \
    --f-min 1000 \
    --f-max 9500 \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --num-workers 0

echo "Goodbye at $(date)."