#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short
#SBATCH --output=exp/log5/slurm/eval/eval_%j.out
#SBATCH --error=exp/log5/slurm/eval/eval_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

module load CUDA/11.7.0
conda activate py310

echo -e "Evaluating..."

srun torchrun \
--standalone \
--nnodes 1 \
--nproc_per_node 3 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29400 \
src/eval_models.py \
    --data-prefix "${HOME}/data/run6/fake_22050_22050_0.7" \
    --model-path-prefix $1 \
    --transform $2 \
    --num-of-scales $3 \
    --wavelet $4 \
    --power $5 \
    --loss-less $6 \
    --model "lcnn"  \
    --batch-size 128 \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --hop-length 100 \
    --seed 0 \
    --log-scale \
    --mean -13.404 -0.00025377 \
    --std 4.8680 1.0000 \
    --train-gans "fbmelgan" \
    --crosseval-gans "lmelgan" "mbmelgan" "melgan" "hifigan" "waveglow" "pwg" "bigvgan" "bigvganl" "avocodo" "conformer" "jsutmbmelgan" "jsutpwg"

echo -e "Evaluating process finished."
echo "Goodbye at $(date)."