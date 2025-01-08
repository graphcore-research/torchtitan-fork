#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest

from low_bits_training.config_manager import JobConfig
from low_bits_training.utils import (
    is_metrics_rank,
    job_config_to_config_dict,
    get_parallel_dims,
    is_metrics_logging_enabled,
    wandb_logging_mode,
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
    "tb,rank,expected", [(True, 0, True), (False, 0, False), (True, 7, False)]
)
def test__is_metrics_logging_enabled(monkeypatch, tb: bool, rank: int, expected: bool):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = tb
    # Default is rank 0 only.
    assert config.metrics.rank_0_only
    assert is_metrics_logging_enabled(config) == expected


def test__wandb_logging_mode__disabled_by_default():
    config = JobConfig.make_default()
    assert wandb_logging_mode(config) == "disabled"


@pytest.mark.parametrize("mode", ["online", "offline"])
def test__wandb_logging_mode__env_variable_setting(monkeypatch, mode):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WANDB_MODE", mode)

    config = JobConfig.make_default()
    config.metrics.enable_tensorboard = True
    assert is_metrics_logging_enabled(config)
    assert wandb_logging_mode(config) == mode


def test__is_metrics_rank__no_pipeline_parallelism(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")

    config = JobConfig.make_default()
    # Default PP == 1 => RANK 0 is the metric logger.
    assert config.experimental.pipeline_parallel_degree == 1
    assert is_metrics_rank(config)


def test__is_metrics_rank__with_pipeline_parallelism(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")

    config = JobConfig.make_default()
    # PP == 8 => RANK 7 is the metric logger.
    config.experimental.pipeline_parallel_degree = 8
    assert is_metrics_rank(config)
