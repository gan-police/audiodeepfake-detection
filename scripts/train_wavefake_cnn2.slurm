#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/experiments/train_wavefake_cnn_single_dist-%j.out
#SBATCH --error=/home/s6kogase/code/experiments/train_wavefake_cnn_single_dist-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --mem=91600M
#SBATCH --partition=A40short

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310


for i in 0 1 2 3 4
do
  echo "Wavefake experiment no: $i "
  python -m src.train_classifier \
	  --seed $i \
      --realdir "/home/s6kogase/wavefake/data/LJspeech-1.1/wavs" \
      --fakedir "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan" \
      --batch-size 512 \
      --epochs 30 \
      --learning-rate 0.0005 \
      --weight-decay 0.001 \
      --model "testnet" \
      --frame-size 200 \
      --scales 64 \
      --amount 10000 \
      --sample-rate 5000 \
      --max-length 1000 \
      --wavelet "cmor4.6-0.87" \
      --m "With db scaling. Less scales. 3791" \
      --fmin 500 \
      --fmax 2500 \
      --tensorboard
done