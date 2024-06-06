from importlib.util import find_spec
from typing import Optional, Dict, Any
import torch
import warnings

__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    return not (torch.xpu.is_available() or find_spec("torch_xla"))
