#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from low_bits_training.config_manager import JobConfig
from dataclasses import fields

import torchtitan.models as tt_models
from torchtitan.models.llama import ModelArgs


def test_model_args_creation_cli():
    """
    Tests model args overriden by command line interface
    """
    config = JobConfig()
    config.parse_args(
        [
            "--model.name=llama2",
            "--model.flavor=debugmodel",
            "--model.dim=1472",
            "--model.n_layers=100",
        ]
    )

    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = tt_models.models_config[config.model.name][config.model.flavor]
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_env(monkeypatch):
    """
    Tests model args overriden by environment variables
    """
    monkeypatch.setenv("MODEL_DIM", "1472")
    monkeypatch.setenv("MODEL_N_LAYERS", "100")
    config = JobConfig()
    config.parse_args(["--model.name=llama2", "--model.flavor=debugmodel"])

    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = tt_models.models_config[config.model.name][config.model.flavor]
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_both(monkeypatch):
    """
    Tests model args overriden by environment variables and CLI
    """
    monkeypatch.setenv("MODEL_DIM", "1472")
    config = JobConfig()
    config.parse_args(
        [
            "--model.name=llama2",
            "--model.flavor=debugmodel",
            "--model.n_layers=100",
        ]
    )

    # check exactly one override string in model.flavor
    assert "override" in config.model.flavor
    assert config.model.flavor.count("override") == 1

    model_args = tt_models.models_config[config.model.name][config.model.flavor]
    assert model_args.dim == 1472
    assert model_args.n_layers == 100


def test_model_args_creation_flavor_only():
    """
    Tests model args created by config.model.flavor only"""

    config = JobConfig()
    config.parse_args(
        [
            "--model.name=llama3",
            "--model.flavor=1B",
        ]
    )

    # No overrides should return same arguments as in config.model.flavor
    model_args = tt_models.models_config[config.model.name][config.model.flavor]

    # Check if the model_args are the same as in llama3:1B
    assert model_args == tt_models.models_config["llama3"]["1B"]

    # Check override string is not present
    assert "override" not in config.model.flavor


def test_job_config_model_args_synced():
    """
    Tests that JobConfig model arguments have matching ModelArgs attribute
    this test will break when a new attribute is added to ModelArgs by upstream torchtitan.
    To fix it, add the matching argument as a command line option.
    """

    config = JobConfig()
    #
    model_actions = {
        action.dest.split(".")[-1]
        for action in config.parser._actions
        if str(action.dest).startswith("model.")
    }
    model_fields = {field.name for field in fields(ModelArgs())}

    print(model_actions)
    print(model_fields)

    # remove model_actions not consumed by ModelArgs
    model_actions -= {"name", "flavor", "model_weights_only", "tokenizer_path"}

    # remove model_fields not set by JobConfig
    model_fields -= {"vocab_size"}

    assert model_actions == model_fields
