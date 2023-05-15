#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log2/prep/prep_all_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log2/prep/prep_all_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40devel
#SBATCH -x node-02

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing all generators binary classification dataset."
conda activate py310

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 128 \
    --window-size 22050 \
    --sample-rate 22050 \
    --equal-distr \
    --max-samples 1751144850 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --directory "${HOME}/data/fake" \
    --target-dir /home/s6kogase/data/run4/

echo "Goodbye at $(date)."