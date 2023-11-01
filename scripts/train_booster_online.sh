#!/bin/bash
#
#SBATCH -A holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=train
#SBATCH --gres=gpu:4
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=24
#SBATCH --partition booster
#SBATCH --time=06:00:00
#SBATCH --output=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log2/slurm/train/train_%j.out
#SBATCH --error=/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log2/slurm/train/train_%j.err
source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

echo "Got nodes:"
echo $SLURM_JOB_NODELIST
echo "Jobs per node:"
echo $SLURM_JOB_NUM_NODES

echo -e "Training..."

#srun --cpu-bind=none 
torchrun \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29400 \
--nnodes 1 \
--nproc_per_node 1 \
src/train_classifier.py \
    --log-dir "/p/home/jusers/gasenzer1/juwels/project_drive/kgasenzer/audiodeepfakes/logs/log2/" \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.001   \
    --epochs 10 \
    --validation-interval 1 \
    --ckpt-every 1 \
    --data-prefix "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/run1/fake_22050_22050_0.7_$2" \
    --cross-dir "${HOME}/project_drive/kgasenzer/audiodeepfakes/data/run1/" \
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
    --hop-length 100 \
    --log-scale \
    --f-min 1 \
    --f-max 11025 \
    --window-size 22050 \
    --sample-rate 22050 \
    --features none \
    --enable-gs \
    --calc-normalization \
    --ddp

echo -e "Training process finished."
echo "Goodbye at $(date)."