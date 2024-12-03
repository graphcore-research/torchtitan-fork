#!/usr/bin/env python3
"""
Parameter Sweep Script for W&B style configs
Config format:
 {
    "command": "./submit.sh", # Executable to run
    "parameters": { # Single parameter set
        "param.name": {"values": [value1, value2]},
        "another.param": {"values": [value3, value4]}
    }
 }

 Or

 {
    "command": "./submit.sh",
    "parameters": [ # Multiple parameter sets
        {
            "param.name": {"values": [value1, value2]},
            "another.param": {"values": [value3, value4]}
        },
        {
            "param.name": {"values": [value5, value6]},
            "different.param": {"values": [value7]}
        }
    ]
 }
"""

import json
import argparse
from itertools import product
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = {
    "command": "tests/test_titan_sweep.sh",
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


def load_config(config_source=None):
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


def validate_parameters(params):
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a dictionary")
    for param, param_config in params.items():
        if not isinstance(param, str) or "." not in param:
            raise ValueError(f"Invalid parameter name: {param}")
        if not isinstance(param_config, dict) or "values" not in param_config:
            raise ValueError(f"Invalid configuration for parameter: {param}")
        if not isinstance(param_config["values"], list):
            raise ValueError(f"Values must be a list for parameter: {param}")


def validate_config(config):
    if "command" not in config:
        raise ValueError("Missing 'command' in config")
    if not isinstance(config["command"], str):
        raise ValueError("'command' must be a string")
    if "parameters" not in config:
        raise ValueError("Missing 'parameters' in config")

    # Handle both single dict and list of dicts
    param_sets = config["parameters"]
    if isinstance(param_sets, dict):
        validate_parameters(param_sets)
    elif isinstance(param_sets, list):
        for param_set in param_sets:
            validate_parameters(param_set)
    else:
        raise ValueError("'parameters' must be either dict or list of dicts")


def generate_command_combinations(parameters):
    param_names = list(parameters.keys())
    param_values = [parameters[param]["values"] for param in param_names]
    for values in product(*param_values):
        yield [(param_names[i], val) for i, val in enumerate(values)]


def run_command(command, command_list):
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


def main():
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

        # Handle both single dict and list of dicts
        param_sets = config["parameters"]
        if isinstance(param_sets, dict):
            param_sets = [param_sets]

        for param_set in param_sets:
            for combo in generate_command_combinations(param_set):
                run_command(command, combo)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
