import torch
from torch.testing._internal.common_utils import TestCase
import copy

import intel_extension_for_pytorch  # noqa
import pytest

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
    contiguous=True, to_channels_last=False, dtype=torch.float
):
    torch.manual_seed(0)
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
    contiguous=True, to_channels_last=False, dtype=torch.float
):
    torch.manual_seed(0)
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


def invoke_unary(fn, in_plain1, in_plian2, in_block1, in_block2, param=None):
    if param is None:
        out_plain1, out_plain2 = fn(in_plain1), fn(in_plian2)
        with torch.xpu.onednn_layout():
            out_block1, out_block2 = fn(in_block1), fn(in_block2)
    else:
        out_plain1, out_plain2 = fn(in_plain1, param), fn(in_plian2, param)
        with torch.xpu.onednn_layout():
            out_block1, out_block2 = fn(in_block1, param), fn(in_block2, param)
    return out_plain1, out_plain2, out_block1, out_block2


def invoke_unary_nn(fn, in_plain1, in_plian2, in_block1, in_block2):
    nn = fn()
    out_plain1, out_plain2 = nn(in_plain1), nn(in_plian2)
    with torch.xpu.onednn_layout():
        out_block1, out_block2 = nn(in_block1), nn(in_block2)
    return out_plain1, out_plain2, out_block1, out_block2


class TestTorchMethod(TestCase):
    def unary_case(self, fn, param=None):
        is_contiguous = True
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(is_contiguous)
        input_block, input_block_xpu = create_block_format_tensor_4d(is_contiguous)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_unary(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu, param
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # channel last bc test
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(True, True)
        input_block, input_block_xpu = create_block_format_tensor_4d(True, True)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_unary(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu, param
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())
        self.assertTrue(
            output_plain_xpu.is_contiguous(memory_format=torch.channels_last)
        )

        is_contiguous = False
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(is_contiguous)
        input_block, input_block_xpu = create_block_format_tensor_4d(is_contiguous)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_unary(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu, param
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # bf16 test
        if fn not in [torch.nn.functional.mish]:
            is_contiguous = True
            input_plain, input_plain_xpu = create_plain_format_tensor_4d(
                is_contiguous, False, torch.bfloat16
            )
            input_block, input_block_xpu = create_block_format_tensor_4d(
                is_contiguous, False, torch.bfloat16
            )
            (
                output_plain,
                output_plain_xpu,
                output_block,
                output_block_xpu,
            ) = invoke_unary(
                fn, input_plain, input_plain_xpu, input_block, input_block_xpu, param
            )
            self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
            if not fn == torch.round:
                self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
            self.assertEqual(output_plain, output_block)
            self.assertEqual(output_plain, output_plain_xpu.cpu())
            self.assertEqual(output_plain, output_block_xpu.cpu())

        # slice test
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(True)
        input_block, input_block_xpu = create_block_format_tensor_4d(True)
        input_plain = input_plain.transpose(0, 1)
        input_plain_xpu = input_plain_xpu.transpose(0, 1)
        input_block = input_block.transpose(0, 1)
        input_block_xpu = input_block_xpu.transpose(0, 1)
        self.assertEqual(input_block_xpu.is_contiguous(), False)
        output_plain, output_plain_xpu, output_block, output_block_xpu = invoke_unary(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu, param
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), False)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

    def test_exp(self):
        self.unary_case(torch.exp)

    def test_log(self):
        self.unary_case(torch.log)

    def test_round(self):
        self.unary_case(torch.round)

    def test_sqrt(self):
        self.unary_case(torch.sqrt)

    def test_tanh(self):
        self.unary_case(torch.tanh)

    def test_abs(self):
        self.unary_case(torch.abs)

    def test_square(self):
        self.unary_case(torch.square)

    def test_pow(self):
        self.unary_case(torch.pow, param=2)

    def test_elu(self):
        self.unary_case(torch.nn.functional.elu)

    def test_relu(self):
        self.unary_case(torch.nn.functional.relu)

    def test_gelu(self):
        self.unary_case(torch.nn.functional.gelu)

    def test_silu(self):
        self.unary_case(torch.nn.functional.silu)

    def test_mish(self):
        self.unary_case(torch.nn.functional.mish)

    def test_hardswish(self):
        self.unary_case(torch.nn.functional.hardswish)

    def test_hardtanh(self):
        self.unary_case(torch.nn.functional.hardtanh)

    def test_hardsigmoid(self):
        self.unary_case(torch.nn.functional.hardsigmoid)

    def test_sigmoid(self):
        self.unary_case(torch.nn.functional.sigmoid)

    def unary_bwd_case(self, fn, dtype=torch.float, to_channels_last=False):
        torch.manual_seed(0)
        inputs = torch.randn([2, 4, 3, 3], dtype=dtype)
        if to_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        inputs.requires_grad_()
        outputs = fn(to_block_cpu(inputs))
        loss = outputs.mean()
        loss.backward()
        inputs_gcpu = inputs.grad

        torch.manual_seed(0)
        inputs = torch.randn([2, 4, 3, 3], dtype=dtype).xpu()
        if to_channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        inputs.requires_grad_()
        outputs = fn(to_block_dpcpp(inputs))
        loss = outputs.mean()
        loss.backward()
        inputs_gxpu_plain = inputs.grad

        self.assertEqual(torch.xpu.is_onednn_layout(inputs_gxpu_plain), False)
        self.assertEqual(inputs_gcpu, inputs_gxpu_plain.cpu())
        if to_channels_last:
            self.assertTrue(
                inputs_gxpu_plain.is_contiguous(memory_format=torch.channels_last)
            )

        with torch.xpu.onednn_layout():
            torch.manual_seed(0)
            inputs = torch.randn([2, 4, 3, 3], dtype=dtype).xpu()
            inputs.requires_grad_()
            outputs = fn(to_block_dpcpp(inputs))
            outputs.mean().backward()
            inputs_gxpu_block = inputs.grad
            self.assertEqual(torch.xpu.is_onednn_layout(inputs_gxpu_block), True)
        self.assertEqual(inputs_gcpu, inputs_gxpu_block.cpu())

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_relu_bwd(self):
        self.unary_bwd_case(torch.nn.functional.relu)
        self.unary_bwd_case(torch.nn.functional.relu, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.relu, torch.float, True)

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_gelu_bwd(self):
        self.unary_bwd_case(torch.nn.functional.gelu)
        self.unary_bwd_case(torch.nn.functional.gelu, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.gelu, torch.float, True)

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_silu_bwd(self):
        self.unary_bwd_case(torch.nn.functional.silu)
        # self.unary_bwd_case(torch.nn.functional.silu, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.silu, torch.float, True)

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_tanh_bwd(self):
        self.unary_bwd_case(torch.nn.functional.tanh)
        # self.unary_bwd_case(torch.nn.functional.tanh, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.tanh, torch.float, True)

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_elu_bwd(self):
        self.unary_bwd_case(torch.nn.functional.elu)
        self.unary_bwd_case(torch.nn.functional.elu, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.elu, torch.float, True)

    @pytest.mark.skipif(
        not torch.xpu.utils.has_2d_block_array(),
        reason="Failed on ATSM only, will be fixed soon.",
    )
    def test_sigmoid_bwd(self):
        self.unary_bwd_case(torch.nn.functional.sigmoid)
        self.unary_bwd_case(torch.nn.functional.sigmoid, torch.bfloat16)
        self.unary_bwd_case(torch.nn.functional.sigmoid, torch.float, True)


class TestNNMethod(TestCase):
    def unary_case_nn(self, fn):
        is_contiguous = True
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(is_contiguous)
        input_block, input_block_xpu = create_block_format_tensor_4d(is_contiguous)
        (
            output_plain,
            output_plain_xpu,
            output_block,
            output_block_xpu,
        ) = invoke_unary_nn(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

        # channel last bc test
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(True, True)
        input_block, input_block_xpu = create_block_format_tensor_4d(True, True)
        (
            output_plain,
            output_plain_xpu,
            output_block,
            output_block_xpu,
        ) = invoke_unary_nn(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())
        self.assertTrue(
            output_plain_xpu.is_contiguous(memory_format=torch.channels_last)
        )

        is_contiguous = False
        input_plain, input_plain_xpu = create_plain_format_tensor_4d(is_contiguous)
        input_block, input_block_xpu = create_block_format_tensor_4d(is_contiguous)
        (
            output_plain,
            output_plain_xpu,
            output_block,
            output_block_xpu,
        ) = invoke_unary_nn(
            fn, input_plain, input_plain_xpu, input_block, input_block_xpu
        )
        self.assertEqual(torch.xpu.is_onednn_layout(output_plain_xpu), False)
        self.assertEqual(torch.xpu.is_onednn_layout(output_block_xpu), True)
        self.assertEqual(output_plain, output_block)
        self.assertEqual(output_plain, output_plain_xpu.cpu())
        self.assertEqual(output_plain, output_block_xpu.cpu())

    def test_elu(self):
        self.unary_case_nn(torch.nn.ELU)

    def test_relu(self):
        self.unary_case_nn(torch.nn.ReLU)

    def test_gelu(self):
        self.unary_case_nn(torch.nn.GELU)

    def test_silu(self):
        self.unary_case_nn(torch.nn.SiLU)

    def test_mish(self):
        self.unary_case_nn(torch.nn.Mish)

    def test_hardswish(self):
        self.unary_case_nn(torch.nn.Hardswish)

    def test_hardsigmoid(self):
        self.unary_case_nn(torch.nn.Hardsigmoid)

    def test_hardtanh(self):
        self.unary_case_nn(torch.nn.Hardtanh)

    def test_sigmoid(self):
        self.unary_case_nn(torch.nn.Sigmoid)
