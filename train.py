#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#

import torch
import wandb

# First import low_bits_training for MonkeyPatching TorchTitan.
import low_bits_training  # noqa: F401
from torchtitan import train as tt_train

if __name__ == "__main__":
    print("TRAINING go brrrrr!")
    config = tt_train.JobConfig()
    config.parse_args()
    wandb.init(project="low-bits-training", entity="graphcore", mode="online", group="test", config=config, sync_tensorboard=True)
    tt_train.main(config)
    torch.distributed.destroy_process_group()
    wandb.finish()
