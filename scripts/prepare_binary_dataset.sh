#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep_single-%j.out
#SBATCH --error=/home/s6kogase/code/out/prep_single-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

real=("A_ljspeech")
fake=("B_melgan")

# approx 180 GB with 40_000 train samples
python -m src.prepare_dataset \
    --train-size 10000 \
    --test-size 2000 \
    --val-size 1100   \
    --batch-size 1100 \
    --wavelet "cmor4.6-0.87"     \
    --sample-number 8736 \
    --window-size 4368 \
    --scales 224 \
    --sample-rate 8000 \
    --f-min 2000   \
    --f-max 4000    \
    --channels 1    \
    --realdir "${HOME}/data/real/$real" \
    --fakedir "${HOME}/data/fake/$fake" \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."