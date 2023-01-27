#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_cnn
#SBATCH --output=/home/s6kogase/code/out/train_normal_%j.out
#SBATCH --error=/home/s6kogase/code/out/train_normal_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:59
#SBATCH --partition=A40short

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py310
python -m src.direct_train_classifier \
    --batch-size 200 \
    --learning-rate 0.001 \
    --weight-decay 0.000001   \
    --epochs 5 \
    --validation-interval 300    \
    --data-prefix "${HOME}/data/fake_cmor4.6-0.87_8000_8736_4368_224_2000-4000_1_10000_hifigan" \
    --nclasses 2 \
    --seed 0 \
    --tensorboard \
    --model "deeptestnet"  \
    --num-workers 4

echo "Goodbye at $(date)."