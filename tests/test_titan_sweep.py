#!/usr/bin/env python3
#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import List, Dict, Any
import json
import tempfile
import subprocess
import sys
import os
import pytest


@pytest.fixture
def echo_script(tmp_path) -> str:
    """
    Create a temporary shell script that echoes its arguments.
    Args:
        tmp_path: Pytest-provided temporary directory path
    Returns:
        Path to the temporary shell script
    """
    script_path = tmp_path / "echo_script.sh"
    script_content = """#!/bin/bash
echo "$@"
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return str(script_path)


def run_sweep_and_capture(config_dict: Dict[str, Any]) -> List[str]:
    """Run the sweep script with given config and return output lines."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
        json.dump(config_dict, tf)
        config_path = tf.name
    try:
        result = subprocess.run(
            [sys.executable, "scripts/titan_sweep.py", config_path],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split("\n")
    finally:
        os.unlink(config_path)


def test_grid_sweep(echo_script: str) -> None:
    """Test grid method parameter sweep."""
    config = {
        "command": echo_script,
        "method": "grid",
        "parameters": {
            "wandb.project": {"values": ["my_project"]},
            "training.steps": {"values": [1000, 2000]},
            "training.batch_size": {"values": [16, 32, 64]},
        },
    }
    outputs = run_sweep_and_capture(config)
    expected_outputs = [
        "--wandb.project my_project --training.steps 1000 --training.batch_size 16",
        "--wandb.project my_project --training.steps 1000 --training.batch_size 32",
        "--wandb.project my_project --training.steps 1000 --training.batch_size 64",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 16",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 32",
        "--wandb.project my_project --training.steps 2000 --training.batch_size 64",
    ]
    assert sorted(outputs) == sorted(expected_outputs)


def test_zip_sweep(echo_script: str) -> None:
    """Test zip method parameter sweep with a single-value parameter."""
    config = {
        "command": echo_script,
        "method": "zip",
        "parameters": {
            "wandb.project": {"values": ["my_project"]},
            "model.type": {"values": ["a", "b"]},
            "learning.rate": {"values": [0.1, 0.2]},
        },
    }
    outputs = run_sweep_and_capture(config)
    expected_outputs = [
        "--wandb.project my_project --model.type a --learning.rate 0.1",
        "--wandb.project my_project --model.type b --learning.rate 0.2",
    ]
    assert sorted(outputs) == sorted(expected_outputs)


if __name__ == "__main__":
    pytest.main([__file__])
