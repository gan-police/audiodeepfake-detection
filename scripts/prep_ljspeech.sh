#!/bin/bash

source ${HOME}/.bashrc

sbatch scripts/prepare_single_dataset.sh
sbatch scripts/prepare_all_dataset.sh
sbatch scripts/prepare_test_dataset.sh