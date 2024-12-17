#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from typing import List, Protocol, Union

import torch.nn as nn

import torchtitan.float8

from torchtitan.config_manager import JobConfig
from torchtitan.float8 import Float8Handler as TTFloat8Handler
from torchtitan.parallelisms import ParallelDims
from torchtitan.logging import logger


class QuantizationHandler(Protocol):
    """Quantization PyTorch model handler interface."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        """Initialize the low-precision handler from job configuration."""
        ...

    def convert(self, model: nn.Module):
        """Convert the model to a low precision format."""
        ...

    def pre_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        """Low-precision pre-optimizer pass (e.g. reduce statistics)."""
        ...

    def post_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        """Low-precision post-optimizer pass (e.g. scatter statistics)."""
        ...

    # TorchTitan backward compatible interface
    def convert_to_float8_training(self, model: nn.Module):
        return self.convert(model)

    def sync_float8_amax_and_scale_history(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        return self.pre_optimizer_pass(model)

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        return self.post_optimizer_pass(model)


class NoQuantizationHandler(QuantizationHandler):
    """Empty TorchTitan model handler (i.e. no modification to model)."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        pass

    def convert(self, model: nn.Module):
        pass

    def pre_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        pass

    def post_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        pass


class Float8Handler(QuantizationHandler):
    """TorchTitan Float8 handler (with delayed scaling) wrapper."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self._impl = TTFloat8Handler(job_config, parallel_dims)

    def convert(self, model: nn.Module):
        return self._impl.convert_to_float8_training(model)

    def pre_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        return self._impl.sync_float8_amax_and_scale_history(model)

    def post_optimizer_pass(self, model: Union[nn.Module, List[nn.Module]]):
        return self._impl.precompute_float8_dynamic_scale_for_fsdp(model)


def build_quantization_handler(
    job_config: JobConfig, parallel_dims: ParallelDims
) -> QuantizationHandler:
    """Quantization handler factory method, based on the job config."""
    logger.info("Building quantization model handler")
    if job_config.float8.enable_float8_linear:
        return Float8Handler(job_config, parallel_dims)
    # None by default.
    return NoQuantizationHandler(job_config, parallel_dims)


# Monkey patching `Float8Handler` to replace it by a factory method.
torchtitan.float8.Float8Handler = build_quantization_handler
