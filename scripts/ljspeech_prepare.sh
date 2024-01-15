#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=prep
#SBATCH --gres=gpu:4
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=96
#SBATCH --partition booster
#SBATCH --time=12:00:00
#SBATCH --output=./logs/log3/slurm/prepare/prep_%j.out
#SBATCH --error=./logs/log3/slurm/prepare/prep_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

python -m scripts.prepare_ljspeech
