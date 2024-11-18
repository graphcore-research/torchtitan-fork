import argparse
import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import sys

from torchtitan.torchtitan.config_manager import JobConfig as TTJobConfig
from torchtitan.torchtitan.logging import logger


class JobConfig(TTJobConfig):
    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            "--wandb.name", type=str, default="", help="Wandb run name"
        )
        self.parser.add_argument(
            "--wandb.group", type=str, default="", help="Wandb run group name"
        )
        self.parser.add_argument(
            "--wandb.project",
            type=str,
            default="low-bits-training",
            help="Wandb project name",
        )
        self.parser.add_argument(
            "--wandb.mode",
            choices=["online", "offline", "disabled"],
            default="online",
            help="Wandb logging mode",
        )

    def parse_args_from_env_vars(self, args_list) -> argparse.Namespace:
        """
        Parse environment variables matching job config options into namespace
        """
        args_list_as_env_vars = [
            v.upper().replace(".", "_").replace("-", "_") for v in args_list
        ]
        env_var_dict = {}
        for orig_arg, env_arg in zip(args_list, args_list_as_env_vars):
            val = os.environ.get(env_arg, None)
            if val is not None:
                env_var_dict[orig_arg] = val

        return argparse.Namespace(**env_var_dict)

    def parse_args(self, args_list: list = sys.argv[1:]):
        """
        Parse arguments according to the following precedence order

        == Highest ==

        1. Command line overrides (must be first for wandb sweeps)
        2. Environment variables overrides
        3. Config toml at job.config_file
        4. Defaults

        == Lowest ==

        Modified from torchtitan to insert environment variable override
        """
        env_args = self.parse_args_from_env_vars(args_list)
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)

        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(
                    f"Error while loading the configuration file: {config_file}"
                )
                logger.exception(f"Error details: {str(e)}")
                raise e

        # override args dict with env_args
        env_args_dict = self._args_to_two_level_dict(env_args)
        _override(args_dict, env_args_dict)

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        _override(args_dict, cmd_args_dict)

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()


def _override(args_dict, overrides_dict):
    """A generic function for providing overrides from other dicts"""
    for section, section_args in overrides_dict.items():
        for k, v in section_args.items():
            args_dict[section][k] = v
    return args_dict
