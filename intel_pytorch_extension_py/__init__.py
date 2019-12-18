import os
import torch
from .version import __version__
import _torch_ipex as core

core._initialize_aten_bindings()

