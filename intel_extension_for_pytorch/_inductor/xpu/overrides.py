import torch  # noqa


def override_decode_device():
    from .lowering import decode_device
    import torch._inductor.lowering
    torch._inductor.lowering.decode_device = decode_device
    return decode_device
