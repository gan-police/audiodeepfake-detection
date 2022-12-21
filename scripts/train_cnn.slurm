#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_kga
#SBATCH --output=logs/out/train-%j.out
#SBATCH --error=logs/out/train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=91600M

source ~/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

python scripts/train_classifier.py --batch-size 128 --epochs 10 --learning-rate 0.001 --model "regression" --frame-size 100 --amount 3000 --sample-rate 10000 --weight-decay 0.0001 --max-length 5000 --wavelet "shan0.1-0.87" --m "With db scaling." --fmin 3000 --fmax 5000 --tensorboard
