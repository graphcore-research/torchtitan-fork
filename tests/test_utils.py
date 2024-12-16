#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from low_bits_training.config_manager import JobConfig
from low_bits_training.utils import is_metrics_rank


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
