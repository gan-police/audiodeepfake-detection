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
#SBATCH --account=userdef

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

#python scripts/train_classifier.py --batch-size 512 --epochs 30 --learning-rate 0.001 --model "testnet" --frame-size 200 --scales 64 --amount 10000 --sample-rate 5000 --weight-decay 0.001 --max-length 1000 --wavelet "cmor4.6-0.87" --m "With db scaling. Less scales." --fmin 500 --fmax 2500 --tensorboard

python scripts/train_classifier.py --batch-size 512 --epochs 30 --learning-rate 0.001 --model "resnet34" --frame-size 224 --scales 224 --amount 20000 --sample-rate 4000 --weight-decay 0.01 --max-length 3000 --wavelet "cmor4.6-0.87" --m "With db scaling. Less scales. ALL Three channels same!" --fmin 20 --fmax 2000 --tensorboard
