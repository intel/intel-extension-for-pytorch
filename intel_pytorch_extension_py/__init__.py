import os
import torch
from .version import __version__
from .optim import *
from .ops import *
import _torch_ipex as core

core._initialize_aten_bindings()
