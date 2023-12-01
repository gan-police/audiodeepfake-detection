#!/bin/bash

sbatch scripts/train_booster.sh packets fbmelgan 256 sym8 2.0 False 320 0
sbatch scripts/train_booster.sh packets fbmelgan 256 coif8 2.0 False 320 0
sbatch scripts/train_booster.sh packets fbmelgan 256 db8 2.0 False 320 0
sbatch scripts/train_booster.sh packets fbmelgan 256 sym10 2.0 False 320 0
sbatch scripts/train_booster.sh packets fbmelgan 256 sym7 2.0 False 320 0
