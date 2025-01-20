#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest

from low_bits_training.config_manager import JobConfig
from low_bits_training.utils import (
    job_config_to_config_dict,
    get_parallel_dims,
    is_metrics_logging_enabled,
    wandb_logging_mode,
    wandb_group,
)


def test__job_config_to_config_dict__proper_dict_structure():
    config = JobConfig.make_default()
    config_dict = job_config_to_config_dict(config)

    assert isinstance(config_dict, dict)
    assert all([isinstance(v, dict) for v in config_dict.values()])


def test__get_parallel_dims__default_values(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "64")

    config = JobConfig.make_default()
    dims = get_parallel_dims(config)
    assert dims.dp_replicate == 1
    assert dims.dp_shard == 64
    assert dims.tp == 1
    assert dims.pp == 1
    assert dims.world_size == 64
    assert dims.enable_loss_parallel


@pytest.mark.parametrize(
    "tb,mode,rank,local_rank,expected",
    [
        (False, "all", 0, 0, False),
        (False, "rank_0", 0, 0, False),
        (False, "local_rank_0", 0, 0, False),
        (True, "all", 0, 0, True),
        (True, "all", 7, 3, True),
        (True, "rank_0", 7, 7, False),
        (True, "rank_0", 0, 0, True),
        (True, "local_rank_0", 0, 0, True),
        (True, "local_rank_0", 4, 0, True),
    ],
)
def test__is_metrics_logging_enabled__distributed_mode_all(
    monkeypatch, tb: bool, mode: str, rank: int, local_rank: int, expected: bool
):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = tb
    config.metrics.rank_0_only = False
    config.metrics.distributed_mode = mode
    assert is_metrics_logging_enabled(config) == expected


def test__is_metrics_logging_enabled__rank_0__no_pipeline_parallelism(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = True
    config.metrics.rank_0_only = False
    config.metrics.distributed_mode = "rank_0"
    # Default PP == 1 => RANK 0 is the metric logger.
    assert config.experimental.pipeline_parallel_degree == 1
    assert is_metrics_logging_enabled(config)


@pytest.mark.parametrize("mode", ["rank_0", "local_rank_0"])
def test__is_metrics_logging_enabled__with_pipeline_parallelism(monkeypatch, mode):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "7")

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = True
    config.metrics.rank_0_only = False
    config.metrics.distributed_mode = mode
    # PP == 8 => RANK 7 is the metric logger.
    config.experimental.pipeline_parallel_degree = 8
    assert is_metrics_logging_enabled(config)


def test__wandb_logging_mode__disabled_by_default():
    config = JobConfig.make_default()
    assert wandb_logging_mode(config) == "disabled"


@pytest.mark.parametrize("mode", ["online", "offline"])
def test__wandb_logging_mode__env_variable_setting(monkeypatch, mode):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WANDB_MODE", mode)

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = True
    config.metrics.rank_0_only = False
    assert is_metrics_logging_enabled(config)
    assert wandb_logging_mode(config) == mode


def test__wandb_group__proper_default():
    config = JobConfig.make_default()
    config.wandb.name = "test"
    # Default case scenario.
    assert config.wandb.group == ""
    assert wandb_group(config) == "test"
    # Group provided scenario.
    config.wandb.group = "group"
    assert wandb_group(config) == "group"
