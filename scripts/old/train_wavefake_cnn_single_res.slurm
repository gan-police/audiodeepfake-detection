#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/experiments/exp_out/train_cnn_single_res-%A_%a.out
#SBATCH --error=/home/s6kogase/code/experiments/exp_out/train_cnn_single_res-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short
#SBATCH --array=0

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

echo "Wavefake experiment no: $SLURM_ARRAY_TASK_ID "
python -m src.train_classifier \
  --seed $SLURM_ARRAY_TASK_ID \
  --realdir "${HOME}/data/real/A_ljspeech" \
  --fakedir "${HOME}/data/fake/B_melgan" \
  --batch-size 128 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --model "resnet18" \
  --frame-size 224 \
  --scales 224 \
  --amount 650000 \
  --sample-rate 8000 \
  --max-length 2000 \
  --wavelet "cmor4.6-0.97" \
  --m "With db scaling. Less scales. 3862." \
  --fmin 2000 \
  --fmax 4000 \
  --tensorboard \
  --mean -54.95780913579314 \
  --std 14.658592904680793
