#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/experiments/exp_out/train_cnn_single-%A_%a.out
#SBATCH --error=/home/s6kogase/code/experiments/exp_out/train_cnn_single-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
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
  --learning-rate 0.001 \
  --weight-decay 0.001 \
  --model "testnet" \
  --frame-size 200 \
  --scales 64 \
  --amount 50000 \
  --sample-rate 5000 \
  --max-length 1000 \
  --wavelet "cmor4.6-0.97" \
  --m "With db scaling to 120. Less scales. 4085. very little freqs" \
  --fmin 20 \
  --fmax 2500 \
  --tensorboard \
  --mean -44.9065068763566 \
  --std 16.335019217554947
done

#--mean -55.61118572569467 \
#--std 15.073388188092709 \
#--wavelet "shan0.001-0.97" \