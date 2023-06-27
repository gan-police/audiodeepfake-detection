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

datasets=("B_melgan" "C_hifigan"  "D_mbmelgan" "E_fbmelgan" "F_waveglow" "G_pwg" "H_lmelgan" "I_avocodo"  "J_bigvgan"  "K_bigvganl")

for i in {0..9};
do
python -m src.prepare_datasets \
    --train-size 0.7 \
    --test-size 0.2 \
    --val-size 0.1   \
    --batch-size 512 \
    --window-size 22050 \
    --sample-rate 22050 \
    --max-samples 1751144850 \
    --realdir "${HOME}/data/real/A_ljspeech" \
    --fakedir "${HOME}/data/fake/${datasets[${i}]}" \
    --directory "${HOME}/data/fake" \
    --target-dir $1 &
declare "pid${i}"=$!
done

wait $pid0
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
wait $pid7
wait $pid8
wait $pid9

echo "Goodbye at $(date)."