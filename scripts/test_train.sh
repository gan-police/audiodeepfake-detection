#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=A40short
#SBATCH -x node-02
#SBATCH --mem=94GB
#SBATCH --output=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/train_learndeepnet_cwt_fbmelgan_%j.out
#SBATCH --error=/home/s6kogase/work/wavelet-audiodeepfake-detection/out/train_learndeepnet_cwt_fbmelgan_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

module load CUDA/11.7.0
conda activate py310

python -m src.train_classifier \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001   \
    --epochs 10 \
    --validation-interval 300    \
    --data-prefix "${HOME}/data/fake_22050_44100_0.7_fbmelgan" \
    --nclasses 2 \
    --seed 0 \
    --model "lcnn"  \
    --f-min 1 \
    --f-max 11025 \
    --window-size 44100 \
    --num-of-scales 256 \
    --sample-rate 22050 \
    --wavelet cmor100.0-2.0 \
    --features "none" \
    --hop-length 100 \
    --transform stft \
    --num-workers 2 \
    --calc-normalization \
    --flattend-size 512 \
    --pbar

echo "Goodbye at $(date)."