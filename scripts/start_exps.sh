#!/bin/bash

sbatch scripts/train.sh packets fbmelgan 256 sym2 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym3 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym4 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym5 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 sym6 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym7 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym8 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 sym9 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 sym10 2.0 False 320 0

sbatch scripts/train.sh packets fbmelgan 256 db2 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db3 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db4 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db5 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 db6 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db7 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db8 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 db9 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 db10 2.0 False 320 0

sbatch scripts/train.sh packets fbmelgan 256 coif2 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif3 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 coif4 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif5 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif6 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif7 2.0 False 320 1
sbatch scripts/train.sh packets fbmelgan 256 coif8 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif9 2.0 False 320 0
sbatch scripts/train.sh packets fbmelgan 256 coif10 2.0 False 320 0
