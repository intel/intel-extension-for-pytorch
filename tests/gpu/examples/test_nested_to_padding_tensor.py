import torch
import intel_extension_for_pytorch as ipex  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest  # noqa

xpu_device = "xpu"
cpu_device = "cpu"


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    if shape is None:
        shape = []
        for size in tensors[0].shape:
            shape.append(size)
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        shape = [ntensors] + shape
    result = tensors[0].new_zeros(shape)
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_on_empty_tensor(self):
        nt = torch.nested.nested_tensor([], device=xpu_device)
        empty = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(empty, torch.tensor([]))

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_simple(self, device=xpu_device):
        def _run_test(dtype=torch.float):
            t = torch.randn(4, 4, 4, device=device, dtype=dtype)
            ts = list(torch.unbind(t))
            ts[0] = ts[0][:-1]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            print(nt.device)
            for padding_value in (0, 1):
                padded = torch.nested.to_padded_tensor(nt, padding_value)
                print(padded)
                correct_output = t.clone()
                if padding_value == 0:
                    correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
                else:
                    correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

                self.assertEqual(padded, correct_output)
                self.assertEqual(padded.device.type, torch.device(device).type)
                self.assertEqual(padded.dtype, dtype)

        _run_test(torch.float)
        _run_test(torch.float16)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_output_size(self, device=xpu_device):
        def _run_test(dtype):
            t = torch.randn(4, 4, 4, device=device, dtype=dtype)
            output_size = (4, 6, 5)
            ts = list(torch.unbind(t))
            ts[0] = ts[0][:-1]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            for padding_value in (0, 1):
                padded = torch.nested.to_padded_tensor(
                    nt, padding_value, output_size=output_size
                )
                correct_output = (
                    torch.ones(output_size, device=device, dtype=dtype) * padding_value
                )
                correct_output[:4:, :4, :4] = t.clone()
                if padding_value == 0:
                    correct_output[0][3] = torch.zeros_like(correct_output[0][3])
                else:
                    correct_output[0][3] = torch.ones_like(correct_output[0][3])

                self.assertEqual(padded, correct_output)
                self.assertEqual(padded.device.type, torch.device(device).type)
                self.assertEqual(padded.dtype, dtype)

        _run_test(torch.float)
        _run_test(torch.float16)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_dim2(self, device=xpu_device):
        def _run_test(dtype):
            ts = [
                torch.randn(160, device=device, dtype=dtype),
                torch.randn(1240, device=device, dtype=dtype),
                torch.randn(50000, device=device, dtype=dtype),
            ]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            pad = 42
            correct_output = []
            for t in ts:
                next_output = torch.ones_like(ts[2]) * pad
                correct_output.append(next_output)
                next_output[: t.size(0)].copy_(t)
            correct_output = torch.stack(correct_output)
            padded = torch.nested.to_padded_tensor(nt, pad)
            self.assertEqual(padded, correct_output)

        _run_test(torch.float)
        _run_test(torch.float16)
        _run_test(torch.double)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_dim3(self, device=xpu_device):
        def _run_test(dtype):
            ts = [
                torch.randn(16, 21, device=device, dtype=dtype),
                torch.randn(24, 32, device=device, dtype=dtype),
                torch.randn(40, 53, device=device, dtype=dtype),
            ]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            pad = 42
            correct_output = []
            for t in ts:
                next_output = torch.ones_like(ts[2]) * pad
                correct_output.append(next_output)
                next_output[: t.size(0), : t.size(1)].copy_(t)
            correct_output = torch.stack(correct_output)
            padded = torch.nested.to_padded_tensor(nt, pad)
            self.assertEqual(padded, correct_output)

        _run_test(torch.float)
        _run_test(torch.float16)
        _run_test(torch.double)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_dim4(self, device=xpu_device):
        def _run_test(dtype):
            ts = [
                torch.randn(16, 21, 13, device=device, dtype=dtype),
                torch.randn(24, 32, 14, device=device, dtype=dtype),
                torch.randn(40, 53, 16, device=device, dtype=dtype),
            ]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            pad = 42
            correct_output = []
            for t in ts:
                next_output = torch.ones_like(ts[2]) * pad
                correct_output.append(next_output)
                next_output[: t.size(0), : t.size(1), : t.size(2)].copy_(t)
            correct_output = torch.stack(correct_output)
            padded = torch.nested.to_padded_tensor(nt, pad)
            self.assertEqual(padded, correct_output)

        _run_test(torch.float)
        _run_test(torch.float16)
        _run_test(torch.double)

    # For now this tests the functionality of noncontiguous_to_padded_tensor
    # and the error message of to_padded_tensor
    # since to_padded_tensor does not support noncontiguous buffer yet
    @torch.inference_mode()
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_to_padded_tensor_noncontiguous(self, device=xpu_device):
        def _run_test(dtype):
            nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
                (2, 3, 6, 7), device, dtype
            )
            # test noncontiguous_to_padded_tensor functionality
            self.assertEqual(
                torch.nested.to_padded_tensor(nt_contiguous, 0.0),
                noncontiguous_to_padded_tensor(nt_noncontiguous),
            )
            # test to_padded_tensor error message
            self.assertRaisesRegex(
                RuntimeError,
                r"for now to_padded_tensor only supports contiguous nested tensor",
                lambda: torch.nested.to_padded_tensor(nt_noncontiguous, 0.0),
            )

        _run_test(torch.float)
        _run_test(torch.float16)
        _run_test(torch.double)
