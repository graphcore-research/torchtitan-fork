#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import os
from typing import Any, Dict

import wandb
from torchtitan.torchtitan.config_manager import JobConfig
from torchtitan.torchtitan.metrics import _get_metrics_rank
from torchtitan.torchtitan.parallelisms.parallel_dims import ParallelDims


def job_config_to_config_dict(job_config: JobConfig) -> Dict[str, Dict[str, Any]]:
    """
    JobConfig is created from Argument Parser as a two level object.

    This a hacky method of converting JobConfig to a ConfigDict that can
    cleanly be consumed by wandb

    """
    first_level_args = [
        attr for attr in dir(job_config) if not (attr.startswith("_") or "parse" in attr)
    ]
    config_dict = {}
    for arg1 in first_level_args:
        assert hasattr(job_config, arg1)
        config_dict[arg1] = {}
        second_level_config = getattr(job_config, arg1)
        second_level_args = [
            attr for attr in dir(second_level_config) if not attr.startswith("_")
        ]
        for arg2 in second_level_args:
            assert hasattr(second_level_config, arg2)
            config_dict[arg1][arg2] = getattr(second_level_config, arg2)
    return config_dict


def get_parallel_dims(job_config) -> ParallelDims:
    world_size = int(os.environ["WORLD_SIZE"])
    return ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )


def is_metrics_rank(job_config: JobConfig) -> bool:
    """Calls torchtitan's `_get_metrics_rank` to figure out if we are on the
    correct process to collect metrics

    This check is needed for pipelining configurations.
    """
    metrics_rank = _get_metrics_rank(get_parallel_dims(job_config))
    return int(os.environ["RANK"]) == metrics_rank


def is_metrics_logging_enabled(job_config: JobConfig) -> bool:
    """Is (W&B) metrics logging enabled on this process."""
    # Are we on the right rank process, if using rank 0 only?
    enable_wb_logging = job_config.metrics.enable_tensorboard and (
        is_metrics_rank(job_config) or not job_config.metrics.rank_0_only
    )
    return enable_wb_logging


def wandb_logging_mode(job_config: JobConfig) -> str:
    """Returns the W&B mode for the current process.

    Combining `WANDB_MODE` env. variable as well as flags in `job_config`
    to decide whether the process should `online` or `disabled`.
    """
    # By default, online on processes where logging is activated.
    wb_mode = os.getenv("WANDB_MODE", "online")
    # Only forward the default mode on processes logging is enabled.
    return wb_mode if is_metrics_logging_enabled(job_config) else "disabled"


def wandb_init(job_config: JobConfig, project: str, entity: str):
    """Initialize properly W&B for current process and training job."""
    config_dict = job_config_to_config_dict(job_config)

    rank = os.environ["RANK"]
    wb_group_id = os.getenv("WANDB_GROUP_ID", None)
    wb_training_run_id = os.getenv("WANDB_TRAINING_RUN_ID")
    # Standardize naming of the run, based on training name and rank.
    wb_run_name = f"{wb_training_run_id}:rank_{rank}"
    # Additional metadata to record on W&B.
    config_dict["metadata"] = {
        "training_run_id": wb_training_run_id,
        "rank": int(rank),
        "local_rank": int(os.environ["LOCAL_RANK"]),
    }
    # W&B logging mode for this process.
    wb_logging_mode = wandb_logging_mode(job_config)
    return wandb.init(
        project=project,
        entity=entity,
        mode=wb_logging_mode,
        config=config_dict,
        name=wb_run_name,
        group=wb_group_id,
    )
