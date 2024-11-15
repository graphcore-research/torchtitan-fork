#!/bin/bash

# cp /data/$GC_USER/.netrc /home/ubuntu/.netrc # probably not needed if you have WANDB_API_KEY set
. setup.sh
bash install.sh
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./train.py --job.config_file ${CONFIG_FILE}
