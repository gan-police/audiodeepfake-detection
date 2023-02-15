#!/bin/bash

source ${HOME}/.bashrc

# this will start 8 * 5 * 3 = 120 jobs for 8 datasets and 3 wavelets for 5 seeds
for gan in "melgan" "hifigan"  "mbmelgan"  "fbmelgan"  "waveglow" "pwg" "lmelgan" "all"
do
    for wavelet in "cmor4.6-0.87" "cmor3.3-4.17" "shan0.01-0.4"
    do 
        echo "starting trainer on seeds 0-4 on $gan with wavelet $wavelet"
        sbatch scripts/final_direct_train_binary_2D_conv.sh $gan $wavelet \
            --output="/home/s6kogase/code/out/train_learndeepnet_$wavelet_$gan_%A_%a.out" \
            --error="/home/s6kogase/code/out/train_learndeepnet_$wavelet_$gan_%A_%a.err"
    done
done