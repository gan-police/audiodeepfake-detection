#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/prep_single_%A_%a.out
#SBATCH --error=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/prep_single_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=A40short
#SBATCH --array=0-6
#SBATCH -x node-02

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

datasets=("C_hifigan"  "D_mbmelgan"  "E_fbmelgan"  "F_waveglow" "G_pwg" "H_lmelgan")

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 2048 \
    --window-size 11025 \
    --sample-rate 22050 \
    --max-samples 1825177725 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --fakedir "${HOME}/data/fake/${datasets[$SLURM_ARRAY_TASK_ID]}" \
    --directory "${HOME}/data/fake"

echo "Goodbye at $(date)."