import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa
import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")
value_range = 30

class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_mode(self):
        def mode_test_helper(input_list):
            for input_cpu in input_list:
                # test from innerest dim to outerest dim
                for dim in reversed(range(input_cpu.dim())):
                    print('input shape = ', input_cpu.shape)
                    print('input dtype = ', input_cpu.dtype)
                    print('dim = ', dim)
                    output_cpu, output_indices_cpu = torch.mode(input_cpu, dim)
                    input_xpu = input_cpu.detach().to(xpu_device)
                    output_xpu, output_indices_xpu = torch.mode(input_xpu, dim)
                    print('checking mode value:')
                    self.assertEqual(output_cpu, output_xpu.cpu())
                    # TODO: The cpu result indices are error in some cases
                    # Here is the issue link to PyTorch
                    # https://github.com/pytorch/pytorch/issues/90261
                    # So skip testing the indices
                    # print('checking mode indice:')
                    # self.assertEqual(output_indices_cpu, output_indices_xpu.cpu())
                    del input_xpu

        # consider host mem intensity, test dtype one by one to save mem
        for tensor_dtype in [torch.int64, torch.bfloat16, torch.bool]:
            input_list = []
            if tensor_dtype == torch.bool:
                # for bool, we small the size
                value_range = 2
            else:
                value_range = 30
            # one dim
            input_list.append(torch.randint(value_range, (1024,)).to(dtype=tensor_dtype))
            # one dim, large input
            input_list.append(torch.randint(value_range, (4099,)).to(dtype=tensor_dtype))
            # two dim, large input
            input_list.append(torch.randint(value_range, (2048, 2048)).to(dtype=tensor_dtype))
            # two dim, large input, odd number
            input_list.append(torch.randint(value_range, (1025, 1999)).to(dtype=tensor_dtype))
            # three dim, odd number
            input_list.append(torch.randint(value_range, (9, 19, 1999)).to(dtype=tensor_dtype))
            # four dim
            input_list.append(torch.randint(value_range, (15, 127, 8, 128)).to(dtype=tensor_dtype))
            mode_test_helper(input_list)
