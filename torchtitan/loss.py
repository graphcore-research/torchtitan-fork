# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from unit_scaling.scale import scale_fwd


def cross_entropy_loss(pred, labels):
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# TODO: compiling this loss function causes CUDA errors, turning off for now
# cross_entropy_loss = torch.compile(cross_entropy_loss)


def umup_nll_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """NLL for use with unit-scaled model. You must apply log_softmax to the output of the final matmul to use this."""
    pred = pred.flatten(0, 1).float()
    labels = labels.flatten(0, 1).float()
    batch_size, _ = pred.shape
    loss = torch.nn.functional.nll_loss(input, target, reduction="sum")
    return scale_fwd(loss, 1 / batch_size)
