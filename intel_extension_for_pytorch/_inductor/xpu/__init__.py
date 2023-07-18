from .overrides import override_decode_device
from .utils import has_triton


if has_triton():
    override_decode_device()
