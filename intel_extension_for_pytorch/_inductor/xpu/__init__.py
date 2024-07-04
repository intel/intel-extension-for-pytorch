import torch
from ...utils.utils import has_xpu
from .overrides import override_size_asserts
from ._meta_registrations import *


def register_xpu_fusion_to_inductor():
    from .lowering import register_onednn_fusion_ops
    from .fx_passes.fusion import _ipex_fusion_init, _ipex_weight_pack_init

    register_onednn_fusion_ops()
    _ipex_fusion_init()
    _ipex_weight_pack_init()


if torch.xpu._is_compiled() and has_xpu():
    override_size_asserts()
