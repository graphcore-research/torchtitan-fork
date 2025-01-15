#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch

import low_bits_training
from low_bits_training.config_manager import JobConfig
from low_bits_training.optimizer import (
    build_lr_scheduler_lambda,
    linear_warmup_cosine_decay,
    build_lr_schedulers,
)
from torch.optim.lr_scheduler import LambdaLR

import torchtitan.optimizer


def make_test_optimizer():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 256)

        def forward(self, x):
            return self.linear(x)

    model = TinyModel()
    optimizer_kwargs = {"lr": 1.0, "betas": (0.9, 0.95), "weight_decay": 0.1}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    return optimizer


def test__optimizer__default_job_config_lr_scheduler_params():
    job_config = JobConfig.make_default()
    # Not modifying default TorchTitan configuration.
    assert job_config.optimizer.lr_scheduler == "linear_warmup_linear_decay"
    assert job_config.optimizer.lr_scheduler_args == "{}"


def test__build_lr_scheduler_lambda__default_torchtitan_linear_scheduler():
    job_config = JobConfig.make_default()
    lr_lambda = build_lr_scheduler_lambda(job_config)
    assert lr_lambda.func is torchtitan.optimizer.linear_warmup_linear_decay


def test__build_lr_scheduler_lambda__warmup_cosine_scheduler():
    job_config = JobConfig.make_default()
    job_config.optimizer.lr_scheduler = "linear_warmup_cosine_decay"
    lr_lambda = build_lr_scheduler_lambda(job_config)
    assert lr_lambda.func is linear_warmup_cosine_decay


def test__build_lr_schedulers__small_model():
    job_config = JobConfig.make_default()
    optimizer = make_test_optimizer()
    lr_schedulers = build_lr_schedulers([optimizer], job_config)
    assert isinstance(lr_schedulers, low_bits_training.optimizer.TTSchedulersContainer)
    assert len(lr_schedulers.schedulers) == 1
    assert all([isinstance(lr, LambdaLR) for lr in lr_schedulers.schedulers])


def test__linear_warmup_cosine_decay__proper_linear_warmup():
    assert linear_warmup_cosine_decay(100, 10, current_step=0) == 1.0 / 101.0
    assert linear_warmup_cosine_decay(100, 10, current_step=50) == 51.0 / 101.0
    assert linear_warmup_cosine_decay(100, 10, current_step=99) == 100.0 / 101.0
    assert linear_warmup_cosine_decay(100, 10, current_step=100) == 1.0


def test__linear_warmup_cosine_decay__proper_cosine_decay():
    assert linear_warmup_cosine_decay(10, 100, current_step=10, final_scaling=0.1) == 1.0
    assert linear_warmup_cosine_decay(10, 100, current_step=110, final_scaling=0.1) == 0.1
