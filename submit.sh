#!/bin/bash
set -eu
if [ -f .env ]; then
  source .env
fi
export WANDB_NAME="${WANDB_NAME:-$(python scripts/random_name.py)}"
# Check that they have been set will error if they haven't
export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export GC_USER="${GC_USER}"

if ! [ -z "${WANDB_PROJECT:-}" ]; then
  echo "W&B project URL: 'https://wandb.ai/graphcore/${WANDB_PROJECT}'"
fi
echo "W&B Name: '${WANDB_NAME}'"
sbatch multinode_trainer.slurm
