#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/code/out/ig_%j.out
#SBATCH --error=/home/s6kogase/code/out/ig_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40medium
#SBATCH --nodelist=node-01

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.integrated_gradients \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7" \
    --plot-path "./plots/attribution" \
    --target-label 1 \
    --times 5056 \
    --model "learndeepnet"  \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --flattend-size 21888 \
    --f-min 1 \
    --f-max 11025 \
    --stft \
    --gans "melgan" "lmelgan" "mbmelgan" "fbmelgan" "hifigan" "waveglow" "pwg" "all"

echo "Goodbye at $(date)."