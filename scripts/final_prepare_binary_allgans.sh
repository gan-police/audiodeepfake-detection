#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep-all-gans-final-%j.out
#SBATCH --error=/home/s6kogase/code/out/prep-all-gans-final-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40devel

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

python -m src.prep_all \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 1000 \
    --wavelet "cmor4.6-0.87"     \
    --window-size 8000 \
    --sample-rate 16000 \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."