#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=/home/s6kogase/code/out/eval_cross_%j.out
#SBATCH --error=/home/s6kogase/code/out/eval_corss_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=A40short
#SBATCH --nodelist=node-03

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.eval_models

echo "Goodbye at $(date)."