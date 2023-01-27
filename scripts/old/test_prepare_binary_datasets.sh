#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep_binary_all-%A_%a.out
#SBATCH --error=/home/s6kogase/code/out/prep_binary_all-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40devel
#SBATCH --array=0-5

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

real=("A_ljspeech")
datasets=("B_melgan" "C_hifigan"  "D_mbmelgan"  "E_fbmelgan"  "F_waveglow" "G_pwg")

python -m src.prep_ds_before_transform \
    --train-size 10000 \
    --test-size 2000 \
    --val-size 1100   \
    --batch-size 1000 \
    --wavelet "cmor4.6-0.87"     \
    --sample-number 8736 \
    --window-size 4368 \
    --scales 224 \
    --sample-rate 8000 \
    --f-min 2000   \
    --f-max 4000    \
    --channels 1    \
    --direct    \
    --realdir "${HOME}/data/real/$real" \
    --fakedir "${HOME}/data/fake/${datasets[$SLURM_ARRAY_TASK_ID]}" \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."
