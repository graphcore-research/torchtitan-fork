#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Callable, Any, Dict
import functools
import json
import math

import torchtitan.optimizer
from torchtitan.optimizer import build_lr_schedulers as tt_build_lr_schedulers
from torchtitan.optimizer import build_optimizers as tt_build_optimizers

from .config_manager import JobConfig
from .metrics import _logger_model_cache


LRLambdaCallable = Callable[[int, int, int], Any]

# TorchTitan internal LR schedulers container.
TTSchedulersContainer = None

_lr_lambda_collection: Dict[str, LRLambdaCallable] = {
    "linear_warmup_linear_decay": torchtitan.optimizer.linear_warmup_linear_decay,
}
"""Collection of LR lambda functions, to be used with torch.LambdaLR generic learning rate scheduler.

Every callable should follow the signature:
    (warmup_steps, decay_steps, current_step, **kwargs) -> float
"""


def register_lr_lambda(lr_lambda: LRLambdaCallable) -> LRLambdaCallable:
    """Register an LR lambda which can be used in TorchTitan.

    Every LR should follow the signature:
        (warmup_steps, decay_steps, current_step, **kwargs) -> float

    NOTE: keeping the same signature as TorchTitan default linear warmup + decay, just
        adding optional dictionary parameters.
    """
    assert (
        lr_lambda.__name__ not in _lr_lambda_collection
    ), f"`{lr_lambda.__name__}` LR lambda already registered."
    _lr_lambda_collection[lr_lambda.__name__] = lr_lambda
    return lr_lambda


def build_lr_scheduler_lambda(job_config: JobConfig) -> Callable[[int], float]:
    """Build a (single) LR scheduler lambda function.

    Args:
        job_config: Job config, with `lr_scheduler` and `lr_scheduler_args` arguments.
    Returns:
        Callable: current_step -> LR scale
    """
    warmup_steps = int(job_config.training.warmup_steps)
    decay_steps = float(max(1, job_config.training.steps - warmup_steps))

    lr_scheduler_name = job_config.optimizer.lr_scheduler
    lr_scheduler_args = json.loads(job_config.optimizer.lr_scheduler_args)
    assert (
        lr_scheduler_name in _lr_lambda_collection
    ), f"Available LR schedulers: {list(_lr_lambda_collection.keys())}"
    lr_lambda = functools.partial(
        _lr_lambda_collection[lr_scheduler_name],
        warmup_steps,
        decay_steps,
        **lr_scheduler_args,
    )
    return lr_lambda


def build_lr_schedulers(optimizers, job_config: JobConfig):
    """Wrapping TorchTitan LR scheduler factory method, keeping track of the LR scheduler.

    Using Job config `optimizer.lr_scheduler` to choose LR scheduler lambda to use.
    """
    global TTSchedulersContainer
    if TTSchedulersContainer is None:
        # TorchTitan `SchedulersContainer` is an internal class defined inside a function
        # so not directly importable. Hence calling the original function once to extract it.
        # TODO: remove when upgrading TorchTitan (importable class on `main`).
        tt_lr_scheduler = tt_build_lr_schedulers(optimizers, job_config)
        TTSchedulersContainer = type(tt_lr_scheduler)
    # Re-build with proper LR scheduler.
    lr_lambda = build_lr_scheduler_lambda(job_config)
    lr_scheduler = TTSchedulersContainer(optimizers, lr_lambda)
    # Caching for logging purposes.
    _logger_model_cache["lr_scheduler"] = lr_scheduler
    return lr_scheduler


def build_optimizers(model_parts, job_config: JobConfig):
    """Wrapping TorchTitan optimizer factory method, keeping track of the optimizer."""
    optimizer = tt_build_optimizers(model_parts, job_config)
    # Caching for logging purposes.
    _logger_model_cache["optimizer"] = optimizer
    return optimizer


# Patching factory methods to keep track of optimizers & LR schedulers.
torchtitan.optimizer.build_optimizers = build_optimizers
torchtitan.optimizer.build_lr_schedulers = build_lr_schedulers


# =======================
# To add a custom LR scheduler define a function like `linear_warmup_cosine_decay`
@register_lr_lambda
def linear_warmup_cosine_decay(
    warmup_steps: int, decay_steps: int, current_step: int, **kwargs
) -> float:
    """Linear warnup followed by cosine decay LR schedule.

    To be used with PyTorch LambdaLR, i.e. this method is returning
    a multiplicative scale of the maximum learning rate.
    """
    if current_step < warmup_steps:
        # linear warmup. 0-indexed step, hence + 1 adjustments
        current_step += 1
        return float(current_step / (warmup_steps + 1))

    # Cosine schedule.
    current_step -= warmup_steps
    scale = (math.cos(current_step / decay_steps * math.pi) + 1.0) * 0.5
    # Final cosine scaling.
    final_scaling = float(kwargs.get("final_scaling", 0.01))
    scale = scale * (1.0 - final_scaling) + final_scaling
    return scale
