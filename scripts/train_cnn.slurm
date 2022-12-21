#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_kga
#SBATCH --output=/home/s6kogase/code/logs/out/train-%j.out
#SBATCH --error=/home/s6kogase/code/logs/out/train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --mem=91600M

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

python scripts/train_classifier.py --batch-size 256 --epochs 15 --learning-rate 0.001 --model "regression" --frame-size 200 --amount 5000 --sample-rate 8000 --weight-decay 0.01 --max-length 2000 --wavelet "cmor4.6-1.0" --m "With db scaling." --fmin 2000 --fmax 4000 --tensorboard
