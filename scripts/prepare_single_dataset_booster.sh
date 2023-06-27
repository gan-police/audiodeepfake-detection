#!/bin/bash
#
#SBATCH -A holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_single_%j.out
#SBATCH --error=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_single_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."

datasets=("B_melgan" "C_hifigan" "D_mbmelgan" "E_fbmelgan" "F_waveglow" "G_pwg" "H_lmelgan" "I_avocodo"  "J_bigvgan"  "K_bigvganl")

for i in {0..9};
do
python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 22050 \
    --sample-rate 22050 \
    --max-samples 1751144850 \
    --fakedir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/fake/${datasets[${i}]}" \
    --realdir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/real/A_ljspeech" \
    --directory "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/fake" \
    --target-dir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/run1/" &
done

echo "Goodbye at $(date)."
