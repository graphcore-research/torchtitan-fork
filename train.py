#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#

import torch

# First import low_bits_training for MonkeyPatching TorchTitan.
import low_bits_training  # noqa: F401
import low_bits_training.utils
import wandb
from torchtitan import train as tt_train

if __name__ == "__main__":
    print("TRAINING go brrrrr!")
    config = tt_train.JobConfig()
    config.parse_args()
    # Logging in W&B? TODO: custom config flag.
    enable_wb_logging = config.metrics.enable_tensorboard
    if enable_wb_logging:
        config_dict = low_bits_training.utils.job_config_to_config_dict(config)
        wandb.init(
            project="low-bits-training",
            entity="graphcore",
            mode="online",
            config=config_dict,
            sync_tensorboard=True,
        )
    # Main TorchTitan training setup & loop
    tt_train.main(config)
    torch.distributed.destroy_process_group()
    if enable_wb_logging:
        wandb.finish()
