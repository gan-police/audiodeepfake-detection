#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=attribute
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log3/slurm/ig/ig_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/exp/log3/slurm/ig/ig_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310

python -m src.integrated_gradients \
    --plot-path "./plots/attribution" \
    --target-label 1 \
    --times 128 \
    --data-prefix "${HOME}/data/run4/fake_22050_22050_0.7" \
    --model-path-prefix ./exp/log3/models/fake_packetssym8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue \
    --model "lcnn"  \
    --batch-size 128 \
    --wavelet sym8 \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --num-of-scales 256 \
    --sample-rate 22050 \
    --features none \
    --hop-length 100 \
    --seed 3 \
    --transform packets \
    --log-scale \
    --loss-less \
    --power 2.0 \
    --gans "fbmelgan" \
    --pbar

echo "Goodbye at $(date)."