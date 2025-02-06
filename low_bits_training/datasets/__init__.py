#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path
from typing import Dict, Any
import json

from datasets import load_dataset
from torchtitan.logging import logger

# The import format here is important: doing a from import will mean the change
# does not propagate
import torchtitan.torchtitan.datasets.hf_datasets as hf_datasets


def _load_slimpajama_dataset(dataset_path: str):
    """Load slimpajama dataset - the dataset path can be appended with additional dataloading kwargs"""
    dataloading_kwargs = {}
    if ":" in dataset_path:
        dataset_path, dataloading_mode = dataset_path.split(":", maxsplit=1)
        dataloading_kwargs = json.loads(dataloading_mode)
        logger.info(f"Using dataset {dataset_path} with arguments: {dataloading_kwargs}")
        if not isinstance(dataloading_kwargs, dict):
            raise ValueError(
                r"Argument dataset_name must be a supported dataset, "
                "or a supported dataset followed by arguments to HF datasets.load_dataset"
                r' method in JSON format: <dataset_name>:{"stream": true, ...}'
            )
    return load_dataset(dataset_path, name="default", split="train", **dataloading_kwargs)


def _process_slimpajama_text(sample: Dict[str, Any]) -> str:
    """Process slimpajama dataset sample text."""
    return sample["text"]


# FIX path of TorchTitan datasets to absolute.
tt_basedir = Path(__file__).resolve().parents[3] / "torchtitan"
hf_datasets.DATASETS["c4_test"].path = str(tt_basedir / "test/assets/c4_test")

hf_datasets.DATASETS["slimpajama"] = hf_datasets.DatasetConfig(
    path="cerebras/SlimPajama-627B",
    loader=_load_slimpajama_dataset,
    text_processor=_process_slimpajama_text,
)
