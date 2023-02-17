#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=A40short
#SBATCH --nodelist=node-01
#SBATCH --array=0-4
#SBATCH --output=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/train_learndeepnet_stft_melgan_%A_%a.out
#SBATCH --error=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/train_learndeepnet_stft_melgan_%A_%a.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310

python -m src.train_classifier \
    --batch-size 100 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001   \
    --epochs 10 \
    --validation-interval 300    \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7_$1" \
    --nclasses 2 \
    --seed $SLURM_ARRAY_TASK_ID \
    --model "learndeepnet"  \
    --f-min 1 \
    --f-max 11025 \
    --num-of-scales 257 \
    --sample-rate 22050 \
    --flattend-size 65664 \
    --stft \
    --num-workers 2

echo "Goodbye at $(date)."