#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from torchtitan.torchtitan.config_manager import JobConfig
from typing import Dict

def job_config_to_config_dict(job_config: JobConfig) -> Dict[str, Dict[str, str]]:
    """
    JobConfig is created from Argument Parser as a two level object.
    
    This a hacky method of converting JobConfig to a ConfigDict that can
    cleanly be consumed by wandb
    
    """
    first_level_args = [attr for attr in dir(job_config) if not (attr.startswith("_") or "parse" in attr)]
    config_dict = {}
    for arg1 in first_level_args:
        assert hasattr(job_config, arg1)
        config_dict[arg1] = {}
        second_level_config = getattr(job_config, arg1)
        second_level_args = [attr for attr in dir(second_level_config) if not attr.startswith("_")]
        for arg2 in second_level_args:
            assert hasattr(second_level_config, arg2)
            config_dict[arg1][arg2] = getattr(second_level_config, arg2)
    return config_dict
