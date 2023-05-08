#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/prep_all_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/prep_all_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=A40short
#SBATCH -x node-02

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing allgans binary classification dataset."
conda activate py310

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 22050 \
    --sample-rate 22050 \
    --max-samples 1598536800 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --leave-out ${HOME}/data/fake/B_melgan ${HOME}/data/fake/C_hifigan ${HOME}/data/fake/D_mbmelgan ${HOME}/data/fake/F_waveglow ${HOME}/data/fake/G_pwg ${HOME}/data/fake/H_lmelgan ${HOME}/data/fake/I_avocodo \
    --directory "${HOME}/data/fake"

echo "Goodbye at $(date)."