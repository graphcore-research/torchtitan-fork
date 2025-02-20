import torch
from collections import defaultdict
import math

def create_umup_adamw(parameters, **optimizer_kwargs):

    param_groups_by_inverse_scale = defaultdict(list)

    base_lr = optimizer_kwargs['lr']
    base_weight_decay = optimizer_kwargs['weight_decay']
    
    for param in parameters:

        if param.lr_scale_formula == "1/sqrt(shape[1])":
            if len(param.shape) != 2:
                raise ValueError("If LR scaling rule is 1/sqrt(fan-in), parameter tensor must have rank 2.")
            param_groups_by_inverse_scale[math.sqrt(param.shape[1])].append(param)
        elif param.lr_scale_formula == "1":
            param_groups_by_inverse_scale[1.0].append(param)
        else:
            raise ValueError(
                f"Invalid LR scaling rule {param.lr_scale_formula} "
                "(valid rules: 1/sqrt(shape[1])), 1"
            )

    # Weight decay is scaled for *independent* AdamW as in Wortsman et al 2023 (https://arxiv.org/abs/2309.14322)
    param_groups = [
        {
            "params": params,
            "lr": base_lr / lr_inverse_scale,
            "weight_decay": base_weight_decay / (base_lr / lr_inverse_scale),
            **optimizer_kwargs
        }
        for lr_inverse_scale, params in param_groups_by_inverse_scale.items()
    ]

    return torch.optim.AdamW(param_groups)
