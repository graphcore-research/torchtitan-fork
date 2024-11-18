#!/bin/bash
set -eu
if [ -f .env ]; then
  source .env
fi
export WANDB_PROJECT=${WANDB_PROJECT:-low-bits-training-dev}
export WANDB_TRAINING_RUN_ID="${WANDB_TRAINING_RUN_ID:-$(python scripts/random_name.py)}"
# Check that they have been set will error if they haven't
export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export GC_USER="${GC_USER}"

echo "W&B project URL: 'https://wandb.ai/graphcore/${WANDB_PROJECT}'"
echo "W&B training run ID: '${WANDB_TRAINING_RUN_ID}'"
sbatch multinode_trainer.slurm
