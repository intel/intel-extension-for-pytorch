from intel_extension_for_pytorch.utils.utils import has_cpu

from . import _tensor_method

if has_cpu():
    from ...cpu.nn import _embeddingbag
    from ...cpu.nn.interaction import interaction, InteractionFunc
    from ...cpu.nn import _roi_align_helper
else:
    _embeddingbag = None
    interaction = None
    InteractionFunc = None
    _roi_align_helper = None
