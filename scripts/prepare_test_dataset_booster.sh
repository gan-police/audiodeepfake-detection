#!/bin/bash
#
#SBATCH -A holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --job-name=prep_ds
#SBATCH --output=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_test_%A_%a.out
#SBATCH --error=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log1/slurm/prep/prep_test_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster

# Prepare all files of all GAN architectures in combination with the real audio
# dataset. The resulting files are resampled but not transformed yet to make
# gradient flow through wavelets possible.

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."
echo "Preparing test binary classification dataset."

datasets=("L_conformer" "M_jsutmbmelgan" "N_jsutpwg")

for i in {0..2};
do
python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 22050 \
    --sample-rate 22050 \
    --max-samples 1751144850 \
    --realdir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/real/A_ljspeech" \
    --testdir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/fake_test/${datasets[${i}]}" \
    --directory "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/fake" \
    --target-dir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/run1/" &
done

echo "Goodbye at $(date)."