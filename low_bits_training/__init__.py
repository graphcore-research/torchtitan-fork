#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
import sys

# Adding torchtitan submodule to `sys.path`
# TODO: check the directory exists?
sys.path.append(os.path.dirname(__file__) + "/../torchtitan")

from . import profiling  # noqa: F401, E402
from . import datasets  # noqa: F401, E402
from . import metrics  # noqa: F401, E402
from . import models  # noqa: F401, E402
from .quantization import (  # noqa: F401, E402
    NoQuantizationHandler,
    Float8Handler,
    QuantizationHandler,
)
from . import optimizer  # noqa: F401, E402

from .config_manager import model_registry as model_registry
