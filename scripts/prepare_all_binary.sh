#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep-all-without_transform-%j.out
#SBATCH --error=/home/s6kogase/code/out/prep-all-without_transform-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=A40short

# Prepare all files of on GAN architecture in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

real=("A_ljspeech")
fake=("C_hifigan")

python -m src.prep_all \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 1000 \
    --wavelet "cmor4.6-0.87"     \
    --window-size 11025 \
    --scales 224 \
    --sample-rate 8000 \
    --f-min 2000   \
    --f-max 4000    \
    --realdir "${HOME}/data/real/$real" \
    --fakedir "${HOME}/data/fake/$fake" \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."