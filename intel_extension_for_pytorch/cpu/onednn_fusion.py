import intel_extension_for_pytorch._C as core


def enable_onednn_fusion(enabled):
    r"""
    Enables or disables oneDNN fusion functionality. If enabled, oneDNN
    operators will be fused in runtime, when intel_extension_for_pytorch
    is imported.

    Args:
        enabled (bool): Whether to enable oneDNN fusion functionality or not.
            Default value is ``True``.

    Examples:

        >>> import intel_extension_for_pytorch as ipex
        >>> # to enable the oneDNN fusion
        >>> ipex.enable_onednn_fusion(True)
        >>> # to disable the oneDNN fusion
        >>> ipex.enable_onednn_fusion(False)
    """

    if enabled:
        core.enable_jit_opt()
    else:
        core.disable_jit_opt()
