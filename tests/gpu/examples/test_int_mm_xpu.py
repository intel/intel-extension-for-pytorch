import torch
import intel_extension_for_pytorch  # noqa
import pytest


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((1024, 512), (512, 1024)),
        ((1023, 513), (513, 1025)),
        ((512, 256), (256, 512)),
        ((513, 255), (255, 511)),
        ((256, 128), (128, 256)),
        ((251, 127), (127, 259)),
        ((128, 64), (64, 128)),
        ((129, 65), (65, 128)),
        ((64, 32), (32, 64)),
        ((63, 31), (31, 67)),
        ((32, 16), (16, 32)),
        ((30, 17), (17, 31)),
        ((16, 8), (8, 16)),
        ((15, 9), (9, 13)),
        ((8, 4), (4, 8)),
        ((7, 7), (7, 3)),
        ((4, 2), (2, 4)),
        ((2, 1), (1, 2)),
    ],
)
def test_int_mm_xpu(shape1, shape2):
    tensor1 = torch.randint(low=-128, high=127, size=shape1, dtype=torch.int8).to("xpu")
    tensor2 = torch.randint(low=-128, high=127, size=shape2, dtype=torch.int8).to("xpu")
    matmul_result = torch.matmul(tensor1.to(torch.float), tensor2.to(torch.float))
    _int_mm_result = torch._int_mm(tensor1, tensor2)
    assert (
        _int_mm_result.dtype == torch.int32
    ), f"Expected dtype torch.int32 but got {_int_mm_result.dtype}"
    assert torch.allclose(
        matmul_result, _int_mm_result.to(torch.float), rtol=1e-05, atol=1e-08
    )


if __name__ == "__main__":
    pytest.main()
