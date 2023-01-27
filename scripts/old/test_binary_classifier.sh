#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=test_res
#SBATCH --output=/home/s6kogase/code/out/test_melgan_testnet_128-%j.out
#SBATCH --error=/home/s6kogase/code/out/test_melgan_testnet_128-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --partition=A40short

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310

python -m src.test_model \
    --batch-size 256 \
    --data-prefix "${HOME}/data/binary_cmor4.6-0.87_8000_8736_4368_224_2000-4000_1_10000_melgan" \
    --nclasses 2 \
    --seed 0 \
    --model "deeptestnet"  \
    --usemodel "${HOME}/code/log/binary_cmor4.6-0.87_8000_8736_4368_224_2000-4000_1_10000_melgan_0.001_5e_deeptestnet_0.pt"   \
    --num-workers 5

echo "Goodbye at $(date)."