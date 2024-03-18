import sys
import torch
import intel_extension_for_pytorch

current_module = sys.modules[__name__]
intel_extension_for_pytorch._register_extension_module("deepspeed", current_module)

from .quantizer import ds_quantize_fp32
