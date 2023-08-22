import intel_extension_for_pytorch._C as core


# API for users to query the capabilities of the IPEX
def has_cpu() -> bool:
    return core._has_cpu()


def has_xpu() -> bool:
    return core._has_xpu()
