#!/bin/bash
NNODES=1                  # TODO: Set total number of nodes
MASTER_ADDR="10.48.93.216"  # TODO: Set master node IP
MASTER_PORT=29500
NODE_RANK=$1                                  

# Check if node rank is provided
if [ -z "$NODE_RANK" ]; then
    echo "Error: NODE_RANK is required. Usage: bash $0 [rank]"
    exit 1
fi

CONFIG="tools/train/config/3_dmd.yaml"
LOGDIR="logs/3_dmd/"
WANDB_SAVE_DIR="wandb"

echo "CONFIG=${CONFIG}"
echo "Starting node with RANK=${NODE_RANK}"

# Create log directory if it doesn't exist
mkdir -p ${LOGDIR}
echo "Log directory ready: ${LOGDIR}"

# Run distributed training
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=8 \
  --rdzv_id=wan_train_job_123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --node_rank=$NODE_RANK \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR > "${LOGDIR}/node_${NODE_RANK}.log" 2>&1