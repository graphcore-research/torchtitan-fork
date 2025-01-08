#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from low_bits_training.quantization import (  # noqa: F401, E402
    NoQuantizationHandler,
    Float8Handler,
    build_quantization_handler,
)
from low_bits_training.utils import get_parallel_dims, JobConfig

import torchtitan.float8


def test__build_quantization_handler__default_handler(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = JobConfig.make_default()
    dims = get_parallel_dims(config)

    qt_handler = build_quantization_handler(config, dims)
    assert isinstance(qt_handler, NoQuantizationHandler)


def test__build_quantization_handler__fp8_handler(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = JobConfig.make_default()
    config.float8.enable_float8_linear = True
    dims = get_parallel_dims(config)

    qt_handler = build_quantization_handler(config, dims)
    assert isinstance(qt_handler, Float8Handler)


def test__build_quantization_handler__tt_monkey_patch():
    assert torchtitan.float8.Float8Handler is build_quantization_handler
