#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
from pathlib import Path
import json

from low_bits_training.config_manager import JobConfig


def test_config_dump(tmp_path: str):
    test_name = "funky-tester-39"
    rank = "2"
    description = "test of the argument parsing"
    os.environ["RANK"] = rank
    os.environ["WANDB_NAME"] = "funky-tester-39"
    config = JobConfig.make_default()
    config.parse_args(
        [f"--job.dump_folder={tmp_path}", f"--job.description={description}"]
    )

    config.dump()

    dump_folder = Path(tmp_path) / test_name

    configs = list(dump_folder.glob("config*rank2*.json"))

    assert (
        len(configs) == 1
    ), f"A config with the expected name was not created by dump {list(dump_folder.iterdir())}"

    loaded_dict = json.loads(configs[0].read_text())
    assert isinstance(loaded_dict, dict)
    assert all([isinstance(v, dict) for v in loaded_dict.values()])
    assert loaded_dict["job"]["description"] == description
