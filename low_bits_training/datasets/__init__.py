#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import os

from torchtitan.datasets.hf_datasets import _supported_datasets

# FIX path of TorchTitan datasets to absolute.
tt_basedir = os.path.abspath(os.path.dirname(__file__) + "../../../torchtitan")
_supported_datasets["c4_test"] = os.path.join(
    tt_basedir, _supported_datasets["c4_test"]
)
