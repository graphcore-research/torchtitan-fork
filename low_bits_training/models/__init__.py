#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
from torchtitan.models import (  # noqa: F401
    model_name_to_cls,
    model_name_to_tokenizer,
    models_config,
)
from torchtitan.models.llama import (  # noqa: F401
    ModelArgs,
    Transformer,
    llama2_configs,
    llama3_configs,
)

# Small Llama3 models
llama3_configs["1B"] = ModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
)
llama3_configs["1B"] = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
)
