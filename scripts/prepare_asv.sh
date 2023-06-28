#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=exp/log5/slurm/prep/prep_single_%A_%a.out
#SBATCH --error=exp/log5/slurm/prep/prep_single_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing single binary classification dataset."
conda activate py310

python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 8000 \
    --sample-rate 16000 \
    --filetype flac \
    --realdir "${HOME}/data/asvspoof/ASVspoof2019/ASVspoof2019_LA_train/audios/A_real" \
    --fakedir "${HOME}/data/asvspoof/ASVspoof2019/ASVspoof2019_LA_train/audios/B_fake" \
    --directory "${HOME}/data/asvspoof/ASVspoof2019/ASVspoof2019_LA_train/audios/" \
    --target-dir $1

echo "Goodbye at $(date)."