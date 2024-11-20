#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import torch

# First import low_bits_training for MonkeyPatching TorchTitan.
import low_bits_training  # noqa: F401
import low_bits_training.utils
import wandb
from low_bits_training.config_manager import JobConfig
from torchtitan import train as tt_train
from torchtitan.logging import logger


def main():
    print("TRAINING go brrrrr!")
    config = JobConfig()
    config.parse_args()

    # W&B init for model metrics & checkpoint.
    low_bits_training.utils.wandb_init(
        job_config=config,
        project=config.wandb.project,
        entity="graphcore",
    )
    # Main TorchTitan training setup & loop
    try:
        tt_train.main(config)
        torch.distributed.destroy_process_group()
    except Exception as e:
        # Error logging before process ends, to record in W&B.
        # Keeping formatting similar to `torchrun` error output.
        logger.error("--- Logging error --")
        logger.error(f"{type(e).__module__}.{type(e).__qualname__}: {e}", exc_info=True)
        raise
    finally:
        # Note keeping W&B init + finish in `main` for clean exception handling.
        wandb.finish()


if __name__ == "__main__":
    main()
