#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/prep_all_%j.out
#SBATCH --error=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/prep_all_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40short
#SBATCH --nodelist=node-01

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing allgans binary classification dataset."
conda activate py310

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 2048 \
    --window-size 11025 \
    --sample-rate 22050 \
    --max-samples 1825177725 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --directory "${HOME}/data/fake"

echo "Goodbye at $(date)."