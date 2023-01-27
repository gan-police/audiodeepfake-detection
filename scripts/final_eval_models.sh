#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/code/out/eval_%j.out
#SBATCH --error=/home/s6kogase/code/out/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models

echo "Goodbye at $(date)."