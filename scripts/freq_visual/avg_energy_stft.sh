#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=avg_energy
#SBATCH --output=./out/avg_energy_%j.out
#SBATCH --error=./out/avg_energy_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short

source "${HOME}/.bashrc"
conda activate py310

python -m scripts.freq_visual.avg_energy_stft