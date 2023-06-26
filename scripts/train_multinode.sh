#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40short
#SBATCH --output=exp/log5/slurm/train/train_lcnn_packets_fbmelgan_%A_%a.out
#SBATCH --error=exp/log5/slurm/train/train_lcnn_packets_fbmelgan_%A_%a.err
#SBATCH --array=3


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

srun torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29400 \
src/train_classifier.py \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.01   \
    --epochs 10 \
    --validation-interval 2 \
    --ckpt-every 1 \
    --data-prefix "${HOME}/data/run6/fake_22050_22050_0.7_$2" \
    --unknown-prefix "${HOME}/data/run6/fake_22050_22050_0.7_all" \
    --nclasses 2 \
    --seed $SLURM_ARRAY_TASK_ID \
    --model lcnn  \
    --transform $1 \
    --num-of-scales $3 \
    --wavelet $4 \
    --power $5 \
    --loss-less $6 \
    --flattend-size $7 \
    --calc-normalization \
    --hop-length 100 \
    --log-scale \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none

echo -e "Training process finished."
echo "Goodbye at $(date)."