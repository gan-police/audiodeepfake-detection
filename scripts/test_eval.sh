#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/eval_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models \
    --data-prefix "${HOME}/data/fake_22050_44100_0.7" \
    --plot-path "./plots/eval/" \
    --model "lcnn"  \
    --batch-size 64 \
    --wavelet cmor100.0-2.0 \
    --f-min 1 \
    --f-max 11025 \
    --window-size 11025 \
    --num-of-scales 256 \
    --sample-rate 22050 \
    --flattend-size 6656 \
    --features "none" \
    --hop-length 50 \
    --seed 0 \
    --transform packets \
    --train-gans "fbmelgan" \
    --crosseval-gans "fbmelgan" "all" "lmelgan" "mbmelgan" "melgan" "hifigan" "waveglow" "pwg" "bigvganl" "avocodo"

echo "Goodbye at $(date)."