#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import os
from typing import List, Optional
import json

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

import torchtitan.datasets.hf_datasets
from torchtitan.datasets.hf_datasets import HuggingFaceDataset, _supported_datasets
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

# FIX path of TorchTitan datasets to absolute.
tt_basedir = os.path.abspath(os.path.dirname(__file__) + "../../../torchtitan")
_supported_datasets = {  # noqa: F811
    "c4_test": os.path.join(tt_basedir, "test/assets/c4_test"),
    "c4": "allenai/c4",
    "slimpajama": "cerebras/SlimPajama-627B",
}


class ModifiedHuggingFaceDataset(HuggingFaceDataset):
    """Modified version of the `HuggingFaceDataset` class.

    The only differences introduced are changes around how `load_dataset()` is called
    to support datasets other than C4, and a new `_supported_datasets` dict with a
    path to the (streamed) `cerebras/SlimPajama-627B` dataset."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        dataloading_kwargs = {}
        if ":" in dataset_name:
            dataset_name, dataloading_mode = dataset_name.split(":", maxsplit=1)
            dataloading_kwargs = json.loads(dataloading_mode)
            if not isinstance(dataloading_kwargs, dict):
                raise ValueError(
                    r"Argument dataset_name must be a supported dataset, "
                    "or a supported dataset followed by arguments to HF datasets.load_dataset"
                    r' method in JSON format: <dataset_name>:{"stream": true, ...}'
                )
        print(f"USING MODIFIED DATASET CLASS. DATASET: {dataset_name=}")
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}"
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}"
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4_test":
            ds = load_dataset(dataset_path, split="train", **dataloading_kwargs)
        else:
            name = "en" if dataset_name == "c4" else "default"
            ds = load_dataset(
                dataset_path, name=name, split="train", **dataloading_kwargs
            )

        # TODO: support shuffling
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []


torchtitan.datasets.hf_datasets.HuggingFaceDataset = ModifiedHuggingFaceDataset
