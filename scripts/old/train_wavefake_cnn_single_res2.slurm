#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/experiments/exp_out/train_cnn_single_res-%A_%a.out
#SBATCH --error=/home/s6kogase/code/experiments/exp_out/train_cnn_single_res-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=A40short
#SBATCH --array=0

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

echo "Wavefake experiment no: $SLURM_ARRAY_TASK_ID "
python -m src.train_classifier \
  --seed $SLURM_ARRAY_TASK_ID \
  --realdir "/home/s6kogase/wavefake/data/LJspeech-1.1/wavs" \
  --fakedir "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan" \
  --batch-size 128 \
  --epochs 30 \
  --learning-rate 0.001 \
  --weight-decay 0.005 \
  --model "resnet18" \
  --frame-size 224 \
  --scales 224 \
  --amount 30000 \
  --sample-rate 16000 \
  --max-length 1000 \
  --wavelet "cmor4.6-0.87" \
  --m "With db scaling. Less scales. 3862." \
  --fmin 6000 \
  --fmax 8000 \
  --tensorboard
done