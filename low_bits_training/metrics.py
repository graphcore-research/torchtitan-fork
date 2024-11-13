#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

import torchtitan
from torchtitan.metrics import JobConfig
from torchtitan.metrics import build_metric_logger as tt_build_metric_logger
from torchtitan.optimizer import build_lr_schedulers as tt_build_lr_schedulers
from torchtitan.optimizer import build_optimizers as tt_build_optimizers
from torchtitan.parallelisms import ParallelDims as ttParallelDims

_logger_model_cache: Dict[str, Any] = {
    "model": None,
    "optimizer": None,
    "lr_scheduler": None,
    "device_mesh": None,
}
"""Logger model (global) cache. A bit hacky, but useful for extracting additional metrics.
"""


class ParallelDims(ttParallelDims):
    def build_mesh(self, device_type) -> DeviceMesh:
        mesh = super().build_mesh(device_type)
        _logger_model_cache["device_mesh"] = mesh
        return mesh


def append_model_metrics(
    metrics: Dict[str, Any], model: Optional[nn.Module]
) -> Dict[str, Any]:
    """Additional model metrics."""
    if model is None:
        return metrics
    return metrics


def append_optimizer_metrics(
    metrics: Dict[str, Any], optimiser: Optional[Any]
) -> Dict[str, Any]:
    """Additional optimizer metrics."""
    if optimiser is None:
        return metrics
    return metrics


def append_lr_scheduler_metrics(
    metrics: Dict[str, Any], lr_scheduler: Optional[Any]
) -> Dict[str, Any]:
    """Additional LR scheduler metrics."""
    if lr_scheduler is None:
        return metrics
    # NOTE: multiple LR schedulers when using pipelining. Last one should be reference.
    metrics["LR"] = lr_scheduler.schedulers[-1].get_last_lr()[0]
    return metrics


def dist_sum(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.SUM.name, group=mesh).item()


def append_total_wps_metrics(metrics: Dict[str, Any], mesh: DeviceMesh):
    if "wps" in metrics:
        metrics["total_wps"] = dist_sum(metrics["wps"], mesh)
    return metrics


class MetricLogger:
    """General metric logger, wrapping TorchTitan one.

    Allowing use captured of model, optimizer and LR scheduler to log additional metrics.
    """

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: Optional[str] = None,
    ):
        self._internal_tt_logger = tt_build_metric_logger(
            job_config, parallel_dims, tag
        )
        self._parallel_dims = parallel_dims
        self._job_config = job_config

    def log(self, metrics: Dict[str, Any], step: int):
        # Additional custom metrics.
        metrics = append_model_metrics(metrics, _logger_model_cache["model"])
        metrics = append_optimizer_metrics(metrics, _logger_model_cache["optimizer"])
        metrics = append_lr_scheduler_metrics(
            metrics, _logger_model_cache["lr_scheduler"]
        )
        metrics = append_total_wps_metrics(metrics, _logger_model_cache["device_mesh"])
        self._internal_tt_logger.log(metrics, step)

    def close(self):
        self._internal_tt_logger.close()


def build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> MetricLogger:
    """Wrapping of TorchTitan `build_metric_logger` factory method, allowing
    for additional features/config.
    """
    return MetricLogger(job_config, parallel_dims, tag)


# Monkey-patching original TorchTitan facotory method.
torchtitan.metrics.build_metric_logger = build_metric_logger


def build_optimizers(model_parts, job_config: JobConfig):
    """Wrapping TorchTitan optimizer factory method, keeping track of the optimizer."""
    optimizer = tt_build_optimizers(model_parts, job_config)
    _logger_model_cache["optimizer"] = optimizer
    return optimizer


def build_lr_schedulers(optimizers, job_config: JobConfig):
    """Wrapping TorchTitan LR scheduler factory method, keeping track of the LR scheduler."""
    lr_scheduler = tt_build_lr_schedulers(optimizers, job_config)
    _logger_model_cache["lr_scheduler"] = lr_scheduler
    return lr_scheduler


# Patching factory methods to keep track of optimizers & LR schedulers.
torchtitan.optimizer.build_optimizers = build_optimizers
torchtitan.optimizer.build_lr_schedulers = build_lr_schedulers
torchtitan.parallelisms.ParallelDims = ParallelDims
