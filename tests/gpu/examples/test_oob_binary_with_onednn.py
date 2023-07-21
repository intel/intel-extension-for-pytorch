import torch
from torch.testing._internal.common_utils import TestCase
import copy

import intel_extension_for_pytorch  # noqa


to_block_cpu_float = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
to_block_dpcpp_float = copy.deepcopy(to_block_cpu_float).xpu()
to_block_cpu_bf = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1).bfloat16()
to_block_dpcpp_bf = copy.deepcopy(to_block_cpu_bf).xpu()
to_block_cpu_hf = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1).half()
to_block_dpcpp_hf = copy.deepcopy(to_block_cpu_hf).xpu()


def to_block_cpu(x, dtype=None):
    t = None
    if dtype is not None:
        t = dtype
    else:
        t = x.dtype
    if t == torch.half:
        return to_block_cpu_hf(x)
    elif t == torch.bfloat16:
        return to_block_cpu_bf(x)
    else:
        return to_block_cpu_float(x)


def to_block_dpcpp(x, dtype=None):
    t = None
    if dtype is not None:
        t = dtype
    else:
        t = x.dtype
    if t == torch.half:
        return to_block_dpcpp_hf(x)
    elif t == torch.bfloat16:
        return to_block_dpcpp_bf(x)
    else:
        return to_block_dpcpp_float(x)


def create_block_format_tensor_4d(
    seed, contiguous=True, to_channels_last=False, dtype=torch.float
):
    torch.manual_seed(seed)
    with torch.xpu.onednn_layout():
        inputs = torch.randn([2, 4, 3, 3], dtype=dtype)
        if to_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        inputs_xpu = inputs.xpu()
        if not contiguous:
            inputs = torch.testing._internal.common_utils.noncontiguous_like(inputs)
            inputs_xpu = torch.testing._internal.common_utils.noncontiguous_like(
                inputs_xpu
            )
        inputs_block = to_block_cpu(inputs, dtype)
        inputs_xpu_block = to_block_dpcpp(inputs_xpu, dtype)
        return inputs_block, inputs_xpu_block


def create_plain_format_tensor_4d(
    seed, contiguous=True, to_channels_last=False, dtype=torch.float
):
    torch.manual_seed(seed)
    inputs = torch.randn([2, 4, 3, 3], dtype=dtype)
    if to_channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)
    inputs_xpu = inputs.xpu()
    if not contiguous:
        inputs = torch.testing._internal.common_utils.noncontiguous_like(inputs)
        inputs_xpu = torch.testing._internal.common_utils.noncontiguous_like(inputs_xpu)
    inputs_block = to_block_cpu(inputs, dtype)
    inputs_xpu_block = to_block_dpcpp(inputs_xpu, dtype)
    return inputs_block, inputs_xpu_block


def invoke_binary(
    fn,
    in_plain_1,
    in_plain_xpu_1,
    in_plain_2,
    in_plain_xpu_2,
    in_block_1,
    in_block_xpu_1,
    in_block_2,
    in_block_xpu_2,
):
    out_plain = fn(in_plain_1, in_plain_2)
    out_plain_xpu = fn(in_plain_xpu_1, in_plain_xpu_2)
    with torch.xpu.onednn_layout():
        out_block = fn(in_block_1, in_block_2)
        out_block_xpu = fn(in_block_xpu_1, in_block_xpu_2)
    return out_plain, out_plain_xpu, out_block, out_block_xpu


class TestTorchMethod(TestCase):
    def assertEqual(self, a, b):
        return super().assertEqual(a, b, atol=1e-3, rtol=1e-3)

    def binary_case(self, fn):
        # test contiguous case
        is_contiguous = True
        input_plain_1, input_plain_xpu_1 = create_plain_format_tensor_4d(
            10, is_contiguous
        )
        input_block_1, input_block_xpu_1 = create_block_format_tensor_4d(
            10, is_contiguous
        )
        input_plain_2, input_plain_xpu_2 = create_plain_format_tensor_4d(
            2, is_contiguous
        )
        input_block_2, input_block_xpu_2 = create_block_format_tensor_4d(
            2, is_contiguous
        )
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_binary(
            fn,
            input_plain_1,
            input_plain_xpu_1,
            input_plain_2,
            input_plain_xpu_2,
            input_block_1,
            input_block_xpu_1,
            input_block_2,
            input_block_xpu_2,
        )
        if not torch.xpu.utils.has_2d_block_array(): 
            # Only ATSM would use marco IPEX_XPU_ONEDNN_LAYOUT to produce block tensor
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # channel last bc test
        input_plain_1, input_plain_xpu_1 = create_plain_format_tensor_4d(10, True, True)
        input_block_1, input_block_xpu_1 = create_block_format_tensor_4d(10, True, True)
        input_plain_2, input_plain_xpu_2 = create_plain_format_tensor_4d(2, True, True)
        input_block_2, input_block_xpu_2 = create_block_format_tensor_4d(2, True, True)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_binary(
            fn,
            input_plain_1,
            input_plain_xpu_1,
            input_plain_2,
            input_plain_xpu_2,
            input_block_1,
            input_block_xpu_1,
            input_block_2,
            input_block_xpu_2,
        )
        if not torch.xpu.utils.has_2d_block_array(): 
            # Only ATSM would use marco IPEX_XPU_ONEDNN_LAYOUT to produce block tensor
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())
        self.assertTrue(
            output_plain_xpu.is_contiguous(memory_format=torch.channels_last)
        )

        # test non-contiguous case
        is_contiguous = False
        input_plain_1, input_plain_xpu_1 = create_plain_format_tensor_4d(
            10, is_contiguous
        )
        input_block_1, input_block_xpu_1 = create_block_format_tensor_4d(
            10, is_contiguous
        )
        input_plain_2, input_plain_xpu_2 = create_plain_format_tensor_4d(
            2, is_contiguous
        )
        input_block_2, input_block_xpu_2 = create_block_format_tensor_4d(
            2, is_contiguous
        )
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_binary(
            fn,
            input_plain_1,
            input_plain_xpu_1,
            input_plain_2,
            input_plain_xpu_2,
            input_block_1,
            input_block_xpu_1,
            input_block_2,
            input_block_xpu_2,
        )
        if not torch.xpu.utils.has_2d_block_array(): 
            # Only ATSM would use marco IPEX_XPU_ONEDNN_LAYOUT to produce block tensor
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # bf16 test
        input_plain_1, input_plain_xpu_1 = create_plain_format_tensor_4d(
            10, True, False, torch.bfloat16
        )
        input_block_1, input_block_xpu_1 = create_block_format_tensor_4d(
            10, True, False, torch.bfloat16
        )
        input_plain_2, input_plain_xpu_2 = create_plain_format_tensor_4d(
            2, True, False, torch.bfloat16
        )
        input_block_2, input_block_xpu_2 = create_block_format_tensor_4d(
            2, True, False, torch.bfloat16
        )
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_binary(
            fn,
            input_plain_1,
            input_plain_xpu_1,
            input_plain_2,
            input_plain_xpu_2,
            input_block_1,
            input_block_xpu_1,
            input_block_2,
            input_block_xpu_2,
        )
        if not torch.xpu.utils.has_2d_block_array(): 
            # Only ATSM would use marco IPEX_XPU_ONEDNN_LAYOUT to produce block tensor
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # slice test
        input_plain_1, input_plain_xpu_1 = create_plain_format_tensor_4d(
            10, True, False, torch.bfloat16
        )
        input_block_1, input_block_xpu_1 = create_block_format_tensor_4d(
            10, True, False, torch.bfloat16
        )
        input_plain_2, input_plain_xpu_2 = create_plain_format_tensor_4d(
            2, True, False, torch.bfloat16
        )
        input_block_2, input_block_xpu_2 = create_block_format_tensor_4d(
            2, True, False, torch.bfloat16
        )
        input_plain_1 = input_plain_1.transpose(2, 3)
        input_plain_xpu_1 = input_plain_xpu_1.transpose(2, 3)
        input_block_1 = input_block_1.transpose(2, 3)
        input_block_xpu_1 = input_block_xpu_1.transpose(2, 3)
        self.assertEqual(input_block_xpu_1.is_contiguous(), False)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_binary(
            fn,
            input_plain_1,
            input_plain_xpu_1,
            input_plain_2,
            input_plain_xpu_2,
            input_block_1,
            input_block_xpu_1,
            input_block_2,
            input_block_xpu_2,
        )
        if not torch.xpu.utils.has_2d_block_array(): 
            # Only ATSM would use marco IPEX_XPU_ONEDNN_LAYOUT to produce block tensor
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), False)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

    def test_add(self):
        self.binary_case(torch.add)

    def test_sub(self):
        self.binary_case(torch.sub)

    def test_div(self):
        # Divisor shall be carefully choosen (not too small)
        self.binary_case(torch.div)

    def test_mul(self):
        self.binary_case(torch.mul)

    def test_max(self):
        self.binary_case(torch.max)

    def test_min(self):
        self.binary_case(torch.min)
