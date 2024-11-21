#!/usr/bin/env python3
"""
Parameter Sweep Script for W&B style configs
Config format:
 {
    "command": "./submit.sh", # Executable to run
    "parameters": {
        "param.name": {"values": [value1, value2]},
        "another.param": {"values": [value3, value4]},
        "flag.param": {"values": [true]}  # Boolean flag parameter
    }
 }
Limitation: You have to use values:[] even for single values
Usage:
 python sweep.py config.json # Load from file
 python sweep.py --config config.json # Load from file
 python sweep.py --config '{"command":"./train.sh", ...}' # JSON string

"""

import json
import argparse
from itertools import product
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = {
    "command": "tests/test_titan_sweep.sh",
    "parameters": {
        "training.batch_size": {"values": [4, 8]},
        "training.seq_len": {"values": [4096, 8192]},
        "profiling.enable_profiling": {"values": [True]},
        "optimiser.lr": {"values": [3e-4]},
        "optimiser.name": {"values": ["AdamW"]},
    },
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


def validate_parameters(config):
    if "command" not in config:
        raise ValueError("Missing 'command' in config")
    if not isinstance(config["command"], str):
        raise ValueError("'command' must be a string")

    parameters = config.get("parameters", {})
    for param, param_config in parameters.items():
        if not isinstance(param, str) or "." not in param:
            raise ValueError(f"Invalid parameter name: {param}")
        if not isinstance(param_config, dict) or "values" not in param_config:
            raise ValueError(f"Invalid configuration for parameter: {param}")
        if not isinstance(param_config["values"], list):
            raise ValueError(f"Values must be a list for parameter: {param}")


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
                # For True values, just add the flag
                cmd.append(f"--{param}")
        else:
            # For string values, add quotes
            if isinstance(value, str):
                cmd.extend([f"--{param}", f'"{value}"'])
            else:
                # For other values (like numbers), add without quotes
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
        validate_parameters(config)
        command = config["command"]
        parameters = config["parameters"]
        for combo in generate_command_combinations(parameters):
            run_command(command, combo)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
