#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log4/slurm/eval/eval_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log4/slurm/eval/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40devel

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models \
    --data-prefix "${HOME}/data/run5/fake_22050_22050_0.7" \
    --model-path-prefix ./exp/log3/models/fake_packetsdb8_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse \
    --model "lcnn"  \
    --batch-size 128 \
    --wavelet db8 \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --num-of-scales 512 \
    --sample-rate 22050 \
    --features none \
    --hop-length 100 \
    --seed 0 1 2 3 4 \
    --transform packets \
    --log-scale \
    --power 2.0 \
    --train-gans "fbmelgan" \
    --crosseval-gans "lmelgan" "mbmelgan" "melgan" "hifigan" "waveglow" "pwg" "bigvgan" "bigvganl" "avocodo" "conformer" "jsutmbmelgan" "jsutpwg"

echo "Goodbye at $(date)."