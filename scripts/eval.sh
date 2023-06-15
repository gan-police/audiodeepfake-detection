#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=exp/log5/slurm/eval/eval_%j.out
#SBATCH --error=exp/log5/slurm/eval/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40devel

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models \
    --data-prefix "${HOME}/data/run6/fake_22050_22050_0.7" \
    --model-path-prefix $1 \
    --transform $2 \
    --num-of-scales $3 \
    --wavelet $4 \
    --power $5 \
    --loss-less $6 \
    --model "lcnn"  \
    --batch-size 128 \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --hop-length 100 \
    --seed 0 1 2 3 4 \
    --log-scale \
    --calc-normalization \
    --train-gans "fbmelgan" \
    --crosseval-gans "lmelgan" "mbmelgan" "melgan" "hifigan" "waveglow" "pwg" "bigvgan" "bigvganl" "avocodo" "conformer" "jsutmbmelgan" "jsutpwg"

echo "Goodbye at $(date)."