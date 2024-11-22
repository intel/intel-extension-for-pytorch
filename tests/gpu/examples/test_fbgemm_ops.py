import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
import fbgemm_gpu  # noqa
import intel_extension_for_pytorch  # noqa


class TestFbGEMMOp(TestCase):
    @parametrize("dtype", [torch.int32, torch.int64])
    def test_asynchronous_complete_cumsum_1d(self, dtype):
        input_cpu = torch.tensor([7, 8, 2, 1, 0, 9, 4], dtype=dtype)
        input_xpu = input_cpu.xpu()
        out_cpu = torch.ops.fbgemm.asynchronous_complete_cumsum(input_cpu)
        out_xpu = torch.ops.fbgemm.asynchronous_complete_cumsum(input_xpu)
        expected_out = torch.tensor([0, 7, 15, 17, 18, 18, 27, 31], dtype=dtype)
        self.assertEqual(expected_out, out_cpu)
        self.assertEqual(expected_out, out_xpu.cpu())

    @parametrize("dtype", [torch.int32, torch.int64])
    def test_asynchronous_complete_cumsum_2d(self, dtype):
        input_cpu = torch.tensor([[7, 8, 2, 1], [5, 2, 0, 7]], dtype=dtype)
        input_xpu = input_cpu.xpu()
        out_cpu = torch.ops.fbgemm.asynchronous_complete_cumsum(input_cpu)
        out_xpu = torch.ops.fbgemm.asynchronous_complete_cumsum(input_xpu)
        expected_out = torch.tensor([[0, 7, 15, 17, 18], [0, 5, 7, 7, 14]], dtype=dtype)
        self.assertEqual(expected_out, out_cpu)
        self.assertEqual(expected_out, out_xpu.cpu())

    @parametrize("dtype", [torch.int64, torch.float, torch.half])
    def test_dense_to_jagged(self, dtype):
        dense_cpu = torch.tensor(
            [[[1, 1], [0, 0], [0, 0]], [[2, 2], [3, 3], [0, 0]]], dtype=dtype
        )
        x_offsets_cpu = torch.tensor([0, 1, 3])
        dense_xpu = dense_cpu.xpu()
        x_offsets_xpu = x_offsets_cpu.xpu()
        out_cpu = torch.ops.fbgemm.dense_to_jagged(dense_cpu, [x_offsets_cpu])
        out_xpu = torch.ops.fbgemm.dense_to_jagged(dense_xpu, [x_offsets_xpu])
        expected_out = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=dtype)
        self.assertEqual(expected_out, out_cpu[0])
        self.assertEqual(expected_out, out_xpu[0].cpu())


instantiate_parametrized_tests(TestFbGEMMOp)

if __name__ == "__main__":
    run_tests()
