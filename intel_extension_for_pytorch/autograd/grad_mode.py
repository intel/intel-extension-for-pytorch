import torch
from torch.autograd.grad_mode import _DecoratorContextManager
import intel_extension_for_pytorch
from typing import Any


class inference_mode(_DecoratorContextManager):
    r"""Context-manager that enables or disables inference mode
    use `torch.inference_mode()` instead after PyTorch 1.9

    """

    def __init__(self, mode=True):
        if not torch._jit_internal.is_scripting():
            super().__init__()
        # Holds a python binding to a RAII guard that can enable or disable
        # inference mode
        self._inference_mode_raii_guard = None
        self.mode = mode

    def __enter__(self):
        self._inference_mode_raii_guard = intel_extension_for_pytorch._C._InferenceMode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del self._inference_mode_raii_guard
