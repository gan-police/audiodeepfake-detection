#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=downsample
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=A100short
#SBATCH --output=exp/log5/slurm/prep/ds_%j.out
#SBATCH --error=exp/log5/slurm/prep/ds_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

module load CUDA/11.7.0
conda activate py310

echo -e "Downsampling..."

python -m src.downsample

echo "Goodbye at $(date)."