#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=A40short
#SBATCH --mem=94GB
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/train_lcnn_packets_bigvganl_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/train_lcnn_packets_bigvganl_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

module load CUDA/11.7.0
conda activate py310

python -m src.train_classifier \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.01   \
    --epochs 15 \
    --validation-interval 300    \
    --data-prefix "${HOME}/data/fake_22050_22050_0.7_fbmelgan" \
    --unknown-prefix "${HOME}/data/fake_22050_22050_0.7_allwithbigvgan" \
    --nclasses 2 \
    --seed 0 \
    --model "lcnn"  \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --num-of-scales 512 \
    --sample-rate 22050 \
    --features "none" \
    --hop-length 100 \
    --transform stft \
    --calc-normalization \
    --num-workers 2 \
    --flattend-size 1024 \
    --pbar

echo "Goodbye at $(date)."