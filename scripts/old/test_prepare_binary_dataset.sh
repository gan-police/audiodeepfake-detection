#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep-without_transform-%j.out
#SBATCH --error=/home/s6kogase/code/out/prep-without_transform-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

real=("A_ljspeech")
fake=("C_hifigan")

python -m src.prep_ds_before_transform \
    --train-size 10000 \
    --test-size 2000 \
    --val-size 1100   \
    --batch-size 1000 \
    --wavelet "cmor4.6-0.87"     \
    --sample-number 16000 \
    --window-size 11025 \
    --scales 150 \
    --sample-rate 16000 \
    --channels 1    \
    --realdir "${HOME}/data/real/$real" \
    --fakedir "${HOME}/data/fake/$fake" \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."