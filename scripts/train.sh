#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40devel
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log3/slurm/train/train_lcnn_packets_fbmelgan_%A_%a.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log3/slurm/train/train_lcnn_packets_fbmelgan_%A_%a.err
#SBATCH --array=0-4

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

module load CUDA/11.7.0
conda activate py310

python -m src.train_classifier \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.1   \
    --epochs 10 \
    --validation-interval 10000    \
    --data-prefix "${HOME}/data/run4/fake_22050_22050_0.7_$2" \
    --unknown-prefix "${HOME}/data/run4/fake_22050_22050_0.7_all" \
    --nclasses 2 \
    --seed $SLURM_ARRAY_TASK_ID \
    --model lcnn  \
    --transform $1 \
    --wavelet $4 \
    --num-of-scales $3 \
    --hop-length 100 \
    --log-scale \
    --f-min 1 \
    --f-max 11025 \
    --power 1.0 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --calc-normalization \
    --num-workers 2 \
    --tensorboard

echo "Goodbye at $(date)."