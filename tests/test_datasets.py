#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path
import pytest

import re
import logging

from low_bits_training.config_manager import JobConfig
import torchtitan.torchtitan.datasets.hf_datasets as hf_datasets
from torchtitan.torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.torchtitan.models import model_name_to_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_custom_dataset_parsing(caplog: pytest.LogCaptureFixture):
    """ """
    job_config = JobConfig()
    c4_test_path = hf_datasets.DATASETS["c4_test"].path
    # TODO: make it so that this test downloads the tokenizer somewhere sensible
    job_config.parse_args(
        [
            "--training.dataset=slimpajama",
            f"--training.dataset_path={c4_test_path}",
            '--training.dataloading_args={"streaming": true}',
            f"--model.tokenizer_path={REPO_ROOT}/torchtitan/torchtitan/datasets/tokenizer/original/tokenizer.model",
        ]
    )
    model_name = job_config.model.name
    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
    with caplog.at_level(logging.INFO):
        hf_datasets.HuggingFaceDataset(
            dataset_name=job_config.training.dataset,
            dataset_path=job_config.training.dataset_path,
            tokenizer=tokenizer,
            seq_len=job_config.training.seq_len,
            world_size=1,
            rank=0,
            infinite=False,
        )

    log_from_custom_loader = re.search("Using dataset .* with arguments:", caplog.text)
    assert (
        log_from_custom_loader
    ), f"The expected log line was not found in: {caplog.text}"
