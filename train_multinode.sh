#!/bin/bash
set -x
# cp /data/$GC_USER/.netrc /home/ubuntu/.netrc # probably not needed if you have WANDB_API_KEY set
# Python venv setup & install repository
. ./scripts/setup.sh
bash ./scripts/install.sh

# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
torchrun --nnodes ${SLURM_JOB_NUM_NODES} --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ${TORCHRUN_EXTRA_ARGS} ./train.py --job.config_file ${CONFIG_FILE} $@
