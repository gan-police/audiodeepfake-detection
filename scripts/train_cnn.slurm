#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_kga
#SBATCH --output=/home/s6kogase/code/logs/out/train-%j.out
#SBATCH --error=/home/s6kogase/code/logs/out/train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=A40devel
#SBATCH --ntasks=1
#SBATCH --mem=91600M

source /home/s6kogase/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)"

conda activate py310

# python -m src.train_classifier --batch-size 512 --epochs 30 --learning-rate 0.001 --model "testnet" --frame-size 200 --scales 64 --amount 10000 --sample-rate 5000 --weight-decay 0.001 --max-length 1000 --wavelet "cmor4.6-0.87" --m "With db scaling. Less scales." --fmin 500 --fmax 2500 --tensorboard

python -m src.train_classifier --batch-size 512 --epochs 15 --learning-rate 0.001 --model "resnet34" --frame-size 224 --scales 224 --amount 15360 --sample-rate 5000 --weight-decay 0.1 --max-length 448 --wavelet "cmor4.6-0.87" --m "With db scaling. All three channels same!" --fmin 20 --fmax 2500 --tensorboard --fakedir "/home/s6kogase/wavefake/data/generated_audio/jsut_parallel_wavegan" --realdir "/home/s6kogase/wavefake/data/LJspeech-1.1/wavs"

# python -m src.train_classifier --batch-size 512 --epochs 15 --learning-rate 0.001 --model "resnet34" --frame-size 224 --scales 224 --amount 10240 --sample-rate 5000 --weight-decay 0.01 --max-length 500 --wavelet "cmor4.6-0.87" --m "With db scaling. All three channels same!" --fmin 20 --fmax 2500 --tensorboard --fakedir "/home/s6kogase/wavefake/data/generated_audio/ljspeech_parallel_wavegan" "/home/s6kogase/wavefake/data/generated_audio/ljspeech_hifiGAN" "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan" "/home/s6kogase/wavefake/data/generated_audio/ljspeech_melgan_large" "/home/s6kogase/wavefake/data/generated_audio/ljspeech_multi_band_melgan" --out-classes 5
