#!/bin/bash
#
#SBATCH -A holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_all_%A_%a.out
#SBATCH --error=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_all_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing all generators binary classification dataset."

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 128 \
    --window-size 22050 \
    --sample-rate 22050 \
    --equal-distr \
    --max-samples 1751144850 \
    --realdir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/real/A_ljspeech" \
    --directory "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/fake" \
    --target-dir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/run1/"

echo "Goodbye at $(date)."