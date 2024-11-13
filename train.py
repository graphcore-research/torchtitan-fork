#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import os

import torch

# First import low_bits_training for MonkeyPatching TorchTitan.
import low_bits_training  # noqa: F401
import low_bits_training.utils
import wandb
from torchtitan import train as tt_train


def main():
    print("TRAINING go brrrrr!")
    config = tt_train.JobConfig()
    config.parse_args()

    # W&B init for model metrics & checkpoint.
    low_bits_training.utils.wandb_init(
        job_config=config,
        project=os.getenv("WANDB_PROJECT", "low-bits-training"),
        entity="graphcore",
    )
    # Main TorchTitan training setup & loop
    tt_train.main(config)
    torch.distributed.destroy_process_group()
    # Note keeping W&B init + finish in `main` for clean exception handling.
    wandb.finish()


if __name__ == "__main__":
    main()
