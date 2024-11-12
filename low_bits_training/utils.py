#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import os
from typing import Dict

from torchtitan.torchtitan.config_manager import JobConfig
from torchtitan.torchtitan.metrics import _get_metrics_rank
from torchtitan.torchtitan.parallelisms.parallel_dims import ParallelDims


def job_config_to_config_dict(job_config: JobConfig) -> Dict[str, Dict[str, str]]:
    """
    JobConfig is created from Argument Parser as a two level object.

    This a hacky method of converting JobConfig to a ConfigDict that can
    cleanly be consumed by wandb

    """
    first_level_args = [
        attr
        for attr in dir(job_config)
        if not (attr.startswith("_") or "parse" in attr)
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


def is_metrics_rank(config) -> bool:
    """Calls torchtitan's `_get_metrics_rank` to figure out if we are on the
    correct process to collect metrics

    This check is needed for pipelining configurations.
    """
    metrics_rank = _get_metrics_rank(get_parallel_dims(config))
    return int(os.environ["RANK"]) == metrics_rank
