#!/bin/bash

source ${HOME}/.bashrc

# this will start 8 * 5 = 40 jobs for 8 datasets for 5 seeds
for gan in "melgan" "hifigan"  "mbmelgan"  "fbmelgan"  "waveglow" "pwg" "lmelgan" "all"
do
    echo "starting trainer on seeds 0-4 on $gan"
    sbatch scripts/train_stft_learndeepnet.sh $gan \
        --output="${HOME}/wavelet-audiodeepfake-detection_code/out/train_learndeepnet_$gan_%A_%a.out" \
        --error="${HOME}/wavelet-audiodeepfake-detection_code/out/train_learndeepnet_$gan_%A_%a.err"
done