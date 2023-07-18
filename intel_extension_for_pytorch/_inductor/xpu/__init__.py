from .overrides import override_decode_device
from .utils import has_triton


if has_triton():
    # Here we have to override decode_device since it is hardcode with CUDA in
    # PyTorch inductor.
    override_decode_device()
