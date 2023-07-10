#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=255
#SBATCH --partition=A100medium
#SBATCH --output=exp/log5/slurm/train/train_%A_%a.out
#SBATCH --error=exp/log5/slurm/train/train_%A_%a.err

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

echo -e "Training..."

torchrun \
--standalone \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29400 \
src/train_classifier.py \
    --log-dir "./exp/log5" \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.001   \
    --epochs 10 \
    --validation-interval 10 \
    --ckpt-every 10 \
    --data-prefix "/home/s6kogase/data/run6/fake_22050_22050_0.7_$2" \
    --cross-dir "/home/s6kogase/data/run6/" \
    --cross-prefix "fake_22050_22050_0.7_" \
    --nclasses 2 \
    --seed 0 \
    --model lcnn  \
    --transform $1 \
    --num-of-scales $3 \
    --wavelet $4 \
    --power $5 \
    --loss-less $6 \
    --flattend-size $7 \
    --aug-contrast \
    --hop-length 100 \
    --log-scale \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --enable-gs \
    --calc-normalization \
    --random-seeds

echo -e "Training process finished."
echo "Goodbye at $(date)."