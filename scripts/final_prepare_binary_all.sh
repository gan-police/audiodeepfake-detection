#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep-single-%A_%a.out
#SBATCH --error=/home/s6kogase/code/out/prep-single-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40devel
#SBATCH --array=0-6

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

real=("A_ljspeech")
datasets=("B_melgan" "C_hifigan"  "D_mbmelgan"  "E_fbmelgan"  "F_waveglow" "G_pwg" "H_lmelgan")

python -m src.prep_all \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 2048 \
    --window-size 8000 \
    --sample-rate 16000 \
    --realdir "${HOME}/data/real/$real" \
    --fakedir "${HOME}/data/fake/${datasets[$SLURM_ARRAY_TASK_ID]}" \
    "${HOME}/data/fake"

echo "Goodbye at $(date)."