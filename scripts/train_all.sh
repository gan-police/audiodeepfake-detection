#!/bin/bash

source ${HOME}/.bashrc

sbatch ./scripts/train.sh packets fbmelgan 256 sym8 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 256 sym8 2.0 False
#sbatch ./scripts/train.sh packets fbmelgan 256 sym8 1.0 True
#sbatch ./scripts/train.sh packets fbmelgan 256 sym8 2.0 True
sbatch ./scripts/train.sh packets fbmelgan 256 db8 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 256 db8 2.0 False
#sbatch ./scripts/train.sh packets fbmelgan 256 db8 1.0 True
#sbatch ./scripts/train.sh packets fbmelgan 256 db8 2.0 True
sbatch ./scripts/train.sh packets fbmelgan 256 sym20 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 256 sym20 2.0 False
#sbatch ./scripts/train.sh packets fbmelgan 256 sym20 1.0 True
#sbatch ./scripts/train.sh packets fbmelgan 256 sym20 2.0 True

# new
sbatch ./scripts/train.sh packets fbmelgan 256 haar 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 256 haar 2.0 False
#sbatch ./scripts/train.sh packets fbmelgan 256 haar 1.0 True
#sbatch ./scripts/train.sh packets fbmelgan 256 haar 2.0 True
# new end

sbatch ./scripts/train.sh stft fbmelgan 256 none 1.0 False
sbatch ./scripts/train.sh stft fbmelgan 256 none 2.0 False


sbatch ./scripts/train.sh packets fbmelgan 512 sym8 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 512 sym8 2.0 False
sbatch ./scripts/train.sh packets fbmelgan 512 db8 1.0 False
sbatch ./scripts/train.sh packets fbmelgan 512 db8 2.0 False
sbatch ./scripts/train.sh packets fbmelgan 512 sym20 1.0 False

sbatch ./scripts/train.sh stft fbmelgan 512 none 1.0 False
sbatch ./scripts/train.sh stft fbmelgan 512 none 2.0 False

