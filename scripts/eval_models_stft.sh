#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/eval_%j.out
#SBATCH --error=/home/s6kogase/wavelet-audiodeepfake-detection_code/out/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40medium
#SBATCH --nodelist=node-01

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models \
    --data-prefix "${HOME}/data/fake_22050_11025_0.7" \
    --plot-path "./plots/eval/" \
    --model "learndeepnet"  \
    --wavelet "cmor3.3-4.17" \
    --f-min 1 \
    --f-max 11025 \
    --num-of-scales 150 \
    --sample-rate 22050 \
    --flattend-size 21888 \
    --stft \
    --train-gans "melgan" "lmelgan" "mbmelgan" "fbmelgan" "hifigan" "waveglow" "pwg" "all" \
    --crosseval-gans "melgan" "lmelgan" "mbmelgan" "fbmelgan" "hifigan" "waveglow" "pwg" "all"

echo "Goodbye at $(date)."