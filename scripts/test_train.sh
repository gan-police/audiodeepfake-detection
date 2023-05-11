#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=A40short
#SBATCH --mem=94GB
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log/train_lcnn_packets_fbmelgan_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log/train_lcnn_packets_fbmelgan_%j.err
#SBATCH -x node-02

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

module load CUDA/11.7.0
conda activate py310

python -m src.train_classifier \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.01   \
    --epochs 10 \
    --validation-interval 396    \
    --data-prefix "${HOME}/data/fake_22050_22050_0.7_$1" \
    --unknown-prefix "${HOME}/data/fake_22050_22050_0.7_allwithjsut" \
    --nclasses 2 \
    --seed 0 \
    --model lcnn  \
    --transform packets \
    --wavelet $2 \
    --num-of-scales $3 \
    --hop-length 100 \
    --log-scale \
    --f-min 1 \
    --f-max 11024 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --calc-normalization \
    --num-workers 2 \
    --pbar

echo "Goodbye at $(date)."