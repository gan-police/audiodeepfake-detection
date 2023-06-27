#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=prep
#SBATCH --output=exp/log5/slurm/prep/prep_%j.out
#SBATCH --error=exp/log5/slurm/prep/prep_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --partition=A100short

source ${HOME}/.bashrc

targetpath="${HOME}/data/run7"

echo "Starting preparation..."
./scripts/prepare_single_dataset.sh ${targetpath} &
pidsingle=$!
./scripts/prepare_test_dataset.sh ${targetpath} &
pidall=$!
./scripts/prepare_all_dataset.sh ${targetpath}

wait $pidsingle
wait $pidall

echo "Cleaning up..."
python scripts/clean_up.py --data-path $targetpath
