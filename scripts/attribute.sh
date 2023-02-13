#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=ig
#SBATCH --output=/home/s6kogase/code/out/ig_%j.out
#SBATCH --error=/home/s6kogase/code/out/ig_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=A40short
#SBATCH --nodelist=node-03

source "${HOME}/.bashrc"
conda activate py310

python -m src.integrated_gradients