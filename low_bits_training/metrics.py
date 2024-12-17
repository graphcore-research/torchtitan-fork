#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

import torchtitan.metrics
import wandb
from torchtitan.optimizer import build_lr_schedulers as tt_build_lr_schedulers
from torchtitan.optimizer import build_optimizers as tt_build_optimizers
from torchtitan.parallelisms import ParallelDims as ttParallelDims

from .config_manager import JobConfig

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
        # Specializing `build_mesh` to cache the device mesh used in training.
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
    # funcol.all_reduce only supporting 1D mesh
    if mesh.size() == 1:
        return x
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.SUM.name, group=mesh).item()


def append_total_wps_metrics(metrics: Dict[str, Any], mesh: DeviceMesh):
    if "wps" in metrics:
        # Summing accross the full mess to get "raw" wps
        total_wps = metrics["wps"]
        if mesh.mesh_dim_names:
            for mesh_dim in mesh.mesh_dim_names:
                total_wps = dist_sum(total_wps, mesh[mesh_dim])
        metrics["total_wps"] = total_wps
    return metrics


class WBMetricLogger:
    """Weight & Biases metric logger, following the same interface as
    TorchTitan Tensorboard `MetricLogger`.

    In addition to sending logs directly to W&B, we capture model,
    optimizer, LR schedule and device mesh to allow additional metrics to be logged.
    """

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: Optional[str] = None,
    ):
        self._parallel_dims = parallel_dims
        self._job_config = job_config

    def log(self, metrics: Dict[str, Any], step: int):
        """Compute additional metrics on top of standard TorchTitan ones + W&B logging."""
        # Additional custom metrics.
        metrics = append_model_metrics(metrics, _logger_model_cache.get("model"))
        metrics = append_optimizer_metrics(metrics, _logger_model_cache.get("optimizer"))
        metrics = append_lr_scheduler_metrics(
            metrics, _logger_model_cache.get("lr_scheduler")
        )
        metrics = append_total_wps_metrics(
            metrics, _logger_model_cache.get("device_mesh")
        )
        # Direct log into W&B.
        # NOTE: no-op if this process W&B has been initialized with `mode=disabled`.
        wandb.log(metrics, step=step)

    def close(self):
        # Clearing the cache, to make sure no reference is held which could mess up with teardown.
        _logger_model_cache.clear()


def build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> WBMetricLogger:
    """Wrapping of TorchTitan `build_metric_logger` factory method, allowing
    to build directly our own W&B metric logger.
    """
    # TODO: keep the option to start Tensorboard one?
    return WBMetricLogger(job_config, parallel_dims, tag)


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
