#!/usr/bin/env python3
#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""
Parameter Sweep Script for W&B style configs
Usage: python titan_sweep.py <json_config_file_or_json_string>

Config format:
 {
    "command": "./submit.sh", # Executable to run
    "method": "grid", # Optional: "grid" (default) or "zip"
    "parameters": { # Single parameter set
        "param.name": {"values": [value1, value2]},
        "another.param": {"values": [value3, value4]}
    }
 }

This will generate the following combinations for grid method:
    ./submit.sh --param.name value1 --another.param value3
    ./submit.sh --param.name value1 --another.param value4
    ./submit.sh --param.name value2 --another.param value3
    ./submit.sh --param.name value2 --another.param value4

And for zip method:
    ./submit.sh --param.name value1 --another.param value3
    ./submit.sh --param.name value2 --another.param value4


Or specify config with multiple parameter sets:

 {
    "command": "./submit.sh",
    "method": "zip", # Optional: "grid" (default) or "zip"
    "parameters": [ # Multiple parameter sets
        {
            "param.name": {"values": [value1, value2]},
            "another.param": {"values": [value3, value4]}
        },
        {
            "project.name": {"values": ["my_project"]},
            "param.name": {"values": [value5, value6]},
            "different.param": {"values": [value7]}
        }
    ]
 }

Special handling for single-value parameters:
If any parameter has a single value, it is added to all combinations in both grid and zip modes.
"""

import json
import argparse
from itertools import product
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union, Iterator, Tuple, Any, Optional, Literal, TypedDict

SearchMethodOptions = Literal["grid", "zip"]

# Type aliases for better readability
ParamConfig = Dict[str, Dict[Literal["values"], List[Any]]]
CommandList = List[Tuple[str, Union[str, int, float, bool]]]


class SweepConfig(TypedDict):
    command: str
    method: SearchMethodOptions
    parameters: List[ParamConfig]


class ParamConfig(TypedDict):
    name: str
    values: List[str | float | int | bool]


DEFAULT_CONFIG: Dict[str, Union[str, List[ParamConfig]]] = {
    "command": "tests/test_titan_sweep.sh",
    "method": "grid",
    "parameters": [
        {
            "training.batch_size": {"values": [4, 8]},
            "training.seq_len": {"values": [4096, 8192]},
            "profiling.enable_profiling": {"values": [True]},
            "optimiser.lr": {"values": [3e-4]},
            "optimiser.name": {"values": ["AdamW"]},
        },
        {
            "training.batch_size": {"values": [16]},
            "training.seq_len": {"values": [4096, 8192]},
        },
    ],
}


def load_config(config_source: Optional[str] = None) -> Dict[str, Any]:
    if not config_source:
        return DEFAULT_CONFIG
    try:
        if Path(config_source).exists():
            with open(config_source) as f:
                return json.load(f)
        return json.loads(config_source)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def validate_parameters(params: ParamConfig, method: str = "grid") -> None:
    """
    Validate a single parameter configuration.

    Args:
        params: Parameter configuration dictionary
        method: Combination method ("grid" or "zip")

    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a dictionary")

    for param, param_config in params.items():
        if not isinstance(param, str) or "." not in param:
            raise ValueError(f"Invalid parameter name: {param}")
        if not isinstance(param_config, dict) or "values" not in param_config:
            raise ValueError(f"Invalid configuration for parameter: {param}")
        if not isinstance(param_config["values"], list):
            raise ValueError(f"Values must be a list for parameter: {param}")

    if method == "zip":
        # Check length consistency excluding single-value parameters
        multi_value_params = {k: v for k, v in params.items() if len(v["values"]) > 1}
        if multi_value_params:
            lengths = [len(param["values"]) for param in multi_value_params.values()]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError(
                    f"For zip method, all multi-value parameters "
                    f"must have {lengths[0]} values each"
                )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the complete configuration.

    Args:
        config: Complete configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    if "command" not in config:
        raise ValueError("Missing 'command' in config")
    if not isinstance(config["command"], str):
        raise ValueError("'command' must be a string")
    if "parameters" not in config:
        raise ValueError("Missing 'parameters' in config")

    method = config.get("method", "grid")
    if method not in ["grid", "zip"]:
        raise ValueError("'method' must be either 'grid' or 'zip'")

    param_sets = config["parameters"]
    if isinstance(param_sets, dict):
        validate_parameters(param_sets, method)
    elif isinstance(param_sets, list):
        for param_set in param_sets:
            validate_parameters(param_set, method)
    else:
        raise ValueError("'parameters' must be either dict or list of dicts")


def generate_command_combinations(
    parameters: ParamConfig, method: SearchMethodOptions = "grid"
) -> Iterator[CommandList]:
    """
    Generate parameter combinations based on the specified method.

    Args:
        parameters: Parameter configuration dictionary
        method: Combination method ("grid" or "zip")

    Yields:
        List of parameter-value tuples for each combination
    """
    # Get single-value parameters first
    single_values = [
        (k, v["values"][0]) for k, v in parameters.items() if len(v["values"]) == 1
    ]

    # Get multi-value parameters
    multi_value_params = {k: v for k, v in parameters.items() if len(v["values"]) > 1}

    if method == "grid":
        if not multi_value_params:
            yield single_values
            return

        param_names = list(multi_value_params.keys())
        param_values = [multi_value_params[param]["values"] for param in param_names]
        for values in product(*param_values):
            yield single_values + [(param_names[i], val) for i, val in enumerate(values)]
    elif method == "zip":
        if not multi_value_params:
            yield single_values
            return

        names = list(multi_value_params.keys())
        for values in zip(*(v["values"] for v in multi_value_params.values())):
            yield single_values + list(zip(names, values))
    else:
        raise ValueError(f"Invalid method: {method}")


def run_command(command: str, command_list: CommandList) -> None:
    """
    Execute the command with the given parameters.

    Args:
        command: Base command to execute
        command_list: List of parameter-value tuples to append to command

    Raises:
        subprocess.CalledProcessError: If command execution fails
    """
    cmd = command.split()
    if isinstance(cmd, str):
        cmd = [command]

    for param, value in command_list:
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{param}")
        else:
            if isinstance(value, str):
                cmd.extend([f"--{param}", f'"{value}"'])
            else:
                cmd.extend([f"--{param}", str(value)])

    try:
        cmdstr = " ".join(cmd)
        subprocess.run(cmdstr, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep script")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("config_file", nargs="?", help="Config file path")
    group.add_argument("--config", help="Config file path or JSON string")

    args = parser.parse_args()
    config_source = args.config if args.config else args.config_file
    config = load_config(config_source)

    try:
        validate_config(config)
        command = config["command"]
        method = config.get("method", "grid")

        param_sets = config["parameters"]
        if isinstance(param_sets, dict):
            param_sets = [param_sets]

        for param_set in param_sets:
            for combo in generate_command_combinations(param_set, method):
                run_command(command, combo)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
