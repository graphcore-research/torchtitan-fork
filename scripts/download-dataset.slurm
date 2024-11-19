#!/bin/bash
#SBATCH --job-name=download_slimpijama
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

export nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export nodes_array=($nodes)
export head_node=${nodes_array[0]}

source setup.sh
echo running on $head_node
python scripts/download_dataset.py
