#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
# Adding torchtitan submodule to `sys.path`
import os
import sys

# TODO: check the directory exists?
sys.path.append(os.path.dirname(__file__) + "/../torchtitan")


from . import datasets  # noqa: F401, E402
from . import models  # noqa: F401, E402
from .low_precision_handler import (  # noqa: F401, E402
    EmptyHandler,
    Float8Handler,
    LowPrecisionHandler,
)
