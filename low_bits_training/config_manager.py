#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import argparse
import hashlib
import json
import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import sys

import torchtitan.models as tt_models

from dataclasses import replace, fields
from typing import Optional, Dict, Any
from torchtitan.config_manager import JobConfig as TTJobConfig
from torchtitan.logging import logger
from torchtitan.models.llama import ModelArgs


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
        # Metrics, distributed mode
        self.parser.add_argument(
            "--metrics.distributed_mode",
            choices=["all", "local_rank_0", "rank_0"],
            default="local_rank_0",
            help="""
                Metrics collection distributed mode.
                    all: All ranks reporting metrics.
                    local_rank_0: Every node local rank 0 only.
                    rank_0: Rank 0 process only.
                When pipeline_parallel_degree is > 1, the option `rank_0` uses the 0th rank of the last stage pipeline group,
                which is the only stage that computes loss metrics.
            """,
        )
        # Optimizer arguments.
        self.parser.add_argument(
            "--optimizer.lr_scheduler",
            type=str,
            default="linear_warmup_linear_decay",
            help="LR scheduler to use",
        )
        self.parser.add_argument(
            "--optimizer.lr_scheduler_args",
            type=str,
            default="{}",
            help="Optional LR scheduler arguments",
        )
        # Training arguments.
        self.parser.add_argument(
            "--training.dataloading_args",
            type=str,
            default=None,
            help=(
                "JSON arguments to pass to W&B load_dataset method when creating the dataset."
                'For example to enable dataset streaming pass: {"streaming": true} (Note the " quotes)'
            ),
        )

        # "llama" as default is not a valid model.name
        self.parser._option_string_actions["--model.name"].default = "llama3"

        self.parser.add_argument(
            "--model.dim",
            type=int,
            help="Model width: dimensionality of token embeddings",
        )
        self.parser.add_argument(
            "--model.n_layers", type=int, help="Number of transformer layers"
        )
        self.parser.add_argument(
            "--model.n_heads",
            type=int,
            help="Number of attention heads per query",
        )
        self.parser.add_argument(
            "--model.n_kv_heads",
            type=Optional[int],
            help="Number of attention heads per key/value pair",
        )
        self.parser.add_argument(
            "--model.multiple_of",
            type=int,
            help="Make SwiGLU hidden layer size multiple of large power of 2",
        )
        self.parser.add_argument(
            "--model.ffn_dim_multiplier",
            type=float,
            help="Expansion factor for MLP hidden layer dimensionality",
        )
        self.parser.add_argument(
            "--model.norm_eps",
            type=float,
            help="Small norm layer denominator stability constant",
        )
        self.parser.add_argument(
            "--model.rope_theta",
            type=float,
            help="Base freqency of rotary positional embeddings",
        )
        self.parser.add_argument(
            "--model.max_seq_len",
            type=int,
            help="Maximum length of input sequences",
        )
        self.parser.add_argument(
            "--model.depth_init",
            type=bool,
            help="Transformer block init scaled by layer ID or total number of layers",
        )

    def combine_args(self):
        """Certain arguments are combined into a single argument for compatibility
        with the Torchtitan training script."""
        if self.training.dataloading_args is not None:
            self.training.dataset = (
                f"{self.training.dataset}:{self.training.dataloading_args}"
            )
        return self

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
        # List all arguments from the parser to get the env var names
        env_args = self.parse_args_from_env_vars([a.dest for a in self.parser._actions])
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
        self.combine_args()
        self._validate_config()

        # If W&B training run name setup, append to output directory.
        if len(self.wandb.name):
            self.job.dump_folder = os.path.join(self.job.dump_folder, self.wandb.name)

        # Override model args
        self.model.flavor = model_registry.override_model_args(
            self.model.name, self.model.flavor, **self.model_args_dict
        )

    @property
    def model_args_dict(self):
        def _get(field):
            field_value = getattr(self.model, field.name)
            field_type = (
                field.type if field.type in [int, float, bool, str] else type(field_value)
            )
            return field_type(field_value) if field_value is not None else None

        return {
            field.name: _get(field)
            for field in fields(ModelArgs())
            if field.name != "vocab_size"
        }

    @classmethod
    def make_default(cls) -> "JobConfig":
        """Create a JobConfig instance with default values."""
        cfg = JobConfig()
        cfg.parse_args([])
        return cfg


def _override(args_dict, overrides_dict):
    """A generic function for providing overrides from other dicts"""
    for section, section_args in overrides_dict.items():
        for k, v in section_args.items():
            args_dict[section][k] = v
    return args_dict


def generate_flavor_hash(base_flavor: str, overrides: Dict[str, Any]) -> str:
    """Generate a deterministic hash for a set of model argument overrides."""
    sorted_items = sorted(overrides.items())
    override_str = json.dumps(sorted_items, sort_keys=True)

    hash_obj = hashlib.sha256(override_str.encode())
    short_hash = hash_obj.hexdigest()[:8]

    return f"{base_flavor}_override_{short_hash}"


class ModelConfigRegistry:
    """Registry for managing dynamic model configurations"""

    def __init__(self):
        self.override_args: Dict[str, Any] = {}

    def override_model_args(self, model_name: str, model_flavor: str, **kwargs):
        """
        Override specific ModelArgs parameters for a given model configuration.
        Returns new flavor name hashed with override args"""
        self._validate(model_name, model_flavor)

        base_config = tt_models.models_config[model_name][model_flavor]
        # prune kwargs if values are same as in base_config or are None
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v != base_config.__dict__[k] and v is not None
        }
        if kwargs:
            new_config = replace(
                base_config, **{k: v for k, v in kwargs.items() if v is not None}
            )
            new_flavor = generate_flavor_hash(model_flavor, kwargs)

            tt_models.models_config[model_name][new_flavor] = new_config

            key = f"{model_name}:{model_flavor}"
            self.override_args[key] = kwargs
            return new_flavor
        else:
            return model_flavor

    def _validate(self, model_name: str, model_flavor: str):
        """Override specific ModelArgs parameters for a given model configuration."""
        if model_name not in tt_models.models_config:
            raise ValueError(f"Unknown model name: {model_name}")
        configs = tt_models.models_config[model_name]
        if model_flavor not in configs:
            raise ValueError(
                f"Unknown model flavor {model_flavor} for model {model_name}"
            )


model_registry = ModelConfigRegistry()
