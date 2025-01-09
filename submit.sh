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

# Checkpointing & log directory + output name (with slurm job ID).
if [ -z "${SBATCH_OUTPUT:-}" ]; then
  export SBATCH_OUTPUT_DIR=${SBATCH_OUTPUT_DIR:-"/data/checkpoints"}
  export SBATCH_OUTPUT="${SBATCH_OUTPUT_DIR}/${WANDB_NAME}/slurm-%j.out"
fi

# Standard naming for slurm job: project + W&B name.
export SBATCH_JOB_NAME="low-bits-training-${WANDB_NAME}"

SLURM_JOB_ID=$(sbatch ${SBATCH_ARGS:-} --parsable multinode_trainer.slurm "$@")
echo "Submitted Slurm batch job ${SLURM_JOB_ID}"
# Update sbatch output env. variable with proper slurm job id.
# NOTE: useful for debugging to have the full & accurate log path in the console!
echo "Slurm output logs: '${SBATCH_OUTPUT/"%j"/${SLURM_JOB_ID}}'"
