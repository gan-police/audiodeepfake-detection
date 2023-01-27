#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/code/out/prep_binary_all-%A_%a.out
#SBATCH --error=/home/s6kogase/code/out/prep_binary_all-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --partition=A40short
#SBATCH --array=0-5

source ${HOME}/.bashrc

echo "Hello on $(hostname) at $(date)."

conda activate py310

real=("A_ljspeech")
datasets=("B_melgan" "C_hifigan"  "D_mbmelgan"  "E_fbmelgan"  "F_waveglow" "G_pwg")

# all would be train 10000 test 2000 val 1100 -> 575 GB
# small might be 2000 400 210 -> 50 GB
python -m src.prepare_dataset \
        --train-size 10000 \
        --test-size 2000 \
        --val-size 1100   \
        --batch-size 2048 \
        --wavelet "cmor4.6-0.87"     \
        --sample-number 8736 \
        --window-size 224 \
        --scales 224 \
        --sample-rate 8000 \
        --f-min 2000   \
        --f-max 4000    \
        --channels 3    \
        --realdir "${HOME}/data/real/$real" \
        --fakedir "${HOME}/data/fake/${datasets[$SLURM_ARRAY_TASK_ID]}" \
        "${HOME}/data/binary"

echo "Goodbye at $(date)."
