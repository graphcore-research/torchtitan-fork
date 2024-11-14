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
# https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
llama3_configs["1B"] = ModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=4,
    multiple_of=256,
    rope_theta=500000,
)  # TODO: share embedding; d_head=64 (default=128) -> 1235746816 params
# https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
llama3_configs["3B"] = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    ffn_dim_multiplier=2.66,  # ffn_dim = 8192
    multiple_of=256,
    rope_theta=500000,
)  # TODO: share embedding -> 3212574720 params
