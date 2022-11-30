import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, dtypes, skipMeta, onlyNativeDeviceTypes
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.utils.dlpack import from_dlpack, to_dlpack
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest

class TestTorchDlPack(TestCase):
    exact_dtype = True

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_capsule_conversion(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(to_dlpack(x))
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_protocol_conversion(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(x)
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_dlpack_shared_storage(self, device):
        x = make_tensor((5,), dtype=torch.float64, device=device)
        z = from_dlpack(to_dlpack(x))
        z[0] = z[0] + 20.0
        self.assertEqual(z, x)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_conversion_with_streams(self, device, dtype):
        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream):
            z = from_dlpack(x)
        stream.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        self.assertEqual(x, y)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack_noncontinguous(self, device, dtype):
        x = make_tensor((25,), dtype=dtype, device=device).reshape(5, 5)
        y1 = x[0]
        y1_dl = torch.from_dlpack(y1)
        self.assertEqual(y1, y1_dl)
        y2 = x[:, 0]
        y2_dl = torch.from_dlpack(y2)
        self.assertEqual(y2, y2_dl)
        y3 = x[1, :]
        y3_dl = torch.from_dlpack(y3)
        self.assertEqual(y3, y3_dl)
        y4 = x[1]
        y4_dl = torch.from_dlpack(y4)
        self.assertEqual(y4, y4_dl)
        y5 = x.t()
        y5_dl = torch.from_dlpack(y5)
        self.assertEqual(y5, y5_dl)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_conversion_with_diff_streams(self, device, dtype):
        stream_a = torch.xpu.Stream()
        stream_b = torch.xpu.Stream()
        with torch.xpu.stream(stream_a):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
            z = torch.from_dlpack(x.__dlpack__(stream_b.xpu_stream))
            stream_a.synchronize()
        stream_b.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack_dtype(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        assert x.dtype == y.dtype

    @skipMeta
    @onlyCUDA
    def test_dlpack_default_stream(self, device):

        class DLPackTensor:

            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, stream=None):
                if torch.version.hip is None:
                    assert stream == 1
                else:
                    assert stream == 0
                capsule = self.tensor.__dlpack__(stream)
                return capsule
        with torch.xpu.stream(torch.xpu.default_stream()):
            x = DLPackTensor(make_tensor((5,), dtype=torch.float32, device=device))
            from_dlpack(x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_tensor_invalid_stream(self, device, dtype):
        with self.assertRaises(TypeError):
            x = make_tensor((5,), dtype=dtype, device=device)
            x.__dlpack__(stream=object())

    @skipMeta
    def test_dlpack_error_on_bool_tensor(self):
        x = torch.tensor([True], dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            to_dlpack(x)

    @skipMeta
    def test_dlpack_export_requires_grad(self):
        x = torch.zeros(10, dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'require gradient'):
            x.__dlpack__()

    @skipMeta
    def test_dlpack_export_is_conj(self):
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, 'conjugate bit'):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_export_non_strided(self):
        x = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, 'strided'):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_normalize_strides(self):
        x = torch.rand(16)
        y = x[::3][:1]
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y.stride(), (3,))
        z = from_dlpack(y)
        self.assertEqual(z.shape, (1,))
        self.assertEqual(z.stride(), (1,))
instantiate_device_type_tests(TestTorchDlPack, globals())
if __name__ == '__main__':
    run_tests()