#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log/prep/prep_single_%A_%a.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log/prep/prep_single_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40devel
#SBATCH --array=11-12
#SBATCH -x node-02

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

datasets=("B_melgan" "C_hifigan"  "D_mbmelgan" "E_fbmelgan" "F_waveglow" "G_pwg" "H_lmelgan" "I_conformer" "J_avocodo" "K_bigvgan" "L_bigvganl" "M_jsutmbmelgan"  "N_jsutpwg")

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 22050 \
    --sample-rate 22050 \
    --max-samples 484570800 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --fakedir "${HOME}/data/fake/${datasets[12]}" \
    --directory "${HOME}/data/fake"

echo "Goodbye at $(date)."