import intel_extension_for_pytorch._C as core


def _is_syngraph_available():
    return core._is_syngraph_available()
