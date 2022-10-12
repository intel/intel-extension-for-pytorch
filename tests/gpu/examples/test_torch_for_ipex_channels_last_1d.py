import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import inspect
import random

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# Note:
# In order to press the gradient of weight below 1,
# the default weight should be set to 1e-ks (ks is kernel_size).
# For now, precision could not be pressed to 1e-5,
# but only if there is a real model which suffers the accuracy problem,
# we won't delve into this issue.


class TestNNMethod(TestCase):
    def test_memory_format(self):
        def test_helper(x):
            y = torch.xpu.to_channels_last_1d(x)
            self.assertFalse(y.is_contiguous())
            self.assertTrue(torch.xpu.is_contiguous_channels_last_1d(y))
            self.assertEqual(y, x)

        test_helper(torch.randn(4, 3, 8))

    def test_memory_format_contiguous_returns_same_tensor_if_already_satisfies(self):
        def test_helper(x):
            alias = torch.xpu.to_channels_last_1d(x)
            alias.fill_(7)
            self.assertEqual(x, alias)

        test_helper(torch.randn(4, 8, 3).permute(0, 2, 1))

    def test_memory_format_preserved_after_permute(self):
        x = torch.randn(4, 3, 8, device=dpcpp_device)
        nwc = torch.xpu.to_channels_last_1d(x)
        y = nwc.permute(0, 2, 1).permute(0, 2, 1)
        self.assertTrue(torch.xpu.is_contiguous_channels_last_1d(y))

    def test_memory_format_consistency(self):
        x = torch.randn(10, 3, 1, 1, device=dpcpp_device)
        x_rep = x.as_strided(x.size(), x.stride())
        self.assertEqual(x.size(), x_rep.size())
        self.assertEqual(x.stride(), x_rep.stride())
        self.assertEqual(x.is_contiguous(), x_rep.is_contiguous())
        self.assertEqual(
            torch.xpu.is_contiguous_channels_last_1d(x), torch.xpu.is_contiguous_channels_last_1d(x_rep))

    def test_memory_format_operators(self):
        def _chunk_op(x, y):
            x1, x2 = x.chunk(2, dim=1)
            return x1 + x2

        def _unsqueeze_op_add(x, y):
            return x[0].unsqueeze(0) + 3

        def _unsqueeze_op_clone(x, y):
            return x[0].unsqueeze(0).clone()

        def _test_helper(x, y, bias):
            return_contig_fns = [
                lambda x, y: y + x,
                lambda x, y: y * x,
                lambda x, y: y.addcdiv(x, y, value=2),
                lambda x, y: y.addcmul(x, y, value=2),
            ]
            bias_fns = [
                lambda x, b: x + b,
                lambda x, b: b + x,
            ]
            fns = [
                lambda x, y: x.clone(),
                lambda x, y: x + 3,
                lambda x, y: 3 * x,
                lambda x, y: x + y,
                lambda x, y: x * y,
                lambda x, y: abs(x),
                lambda x, y: x.abs(),
                lambda x, y: x.abs_(),
                lambda x, y: x.acos(),
                lambda x, y: x.acos_(),
                lambda x, y: x.add(y, alpha=3),
                lambda x, y: x.add_(y, alpha=3),
                lambda x, y: x.addcdiv(y, y, value=2),
                lambda x, y: x.addcdiv_(y, y, value=2),
                lambda x, y: x.addcmul(y, y, value=2),
                lambda x, y: x.addcmul_(y, y, value=2),
                lambda x, y: x.acosh(),
                lambda x, y: x.acosh_(),
                lambda x, y: x.asinh(),
                lambda x, y: x.asinh_(),
                lambda x, y: x.atanh(),
                lambda x, y: x.atanh_(),
                lambda x, y: x.asin(),
                lambda x, y: x.asin_(),
                lambda x, y: x.atan(),
                lambda x, y: x.atan2(y),
                lambda x, y: x.atan2_(y),
                lambda x, y: x.ceil(),
                lambda x, y: x.ceil_(),
                lambda x, y: x.clamp(-1, 1),
                lambda x, y: x.cos(),
                lambda x, y: x.cosh(),
                lambda x, y: x.div(0.5),
                lambda x, y: x.div_(0.5),
                lambda x, y: x.div(y),
                lambda x, y: x.div_(y),
                lambda x, y: x.digamma(),
                lambda x, y: x.digamma_(),
                lambda x, y: x.erf(),
                lambda x, y: x.erfc(),
                lambda x, y: x.erfinv(),
                lambda x, y: x.erfinv_(),
                lambda x, y: x.exp(),
                lambda x, y: x.expm1(),
                lambda x, y: x.expm1_(),
                lambda x, y: x.floor(),
                lambda x, y: x.floor_(),
                lambda x, y: x.fmod(2),
                lambda x, y: x.frac(),
                lambda x, y: x.hypot(y),
                lambda x, y: x.hypot_(y),
                lambda x, y: x.i0(),
                lambda x, y: x.i0_(),
                lambda x, y: x.lerp(y, 0.5),
                lambda x, y: x.log(),
                lambda x, y: x.log_(),
                lambda x, y: x.log10(),
                lambda x, y: x.log10_(),
                lambda x, y: x.log1p(),
                lambda x, y: x.log1p_(),
                lambda x, y: x.log2(),
                lambda x, y: x.log2_(),
                lambda x, y: x.mul(3),
                lambda x, y: x.mul_(3),
                lambda x, y: x.neg(),
                lambda x, y: x.neg_(),
                lambda x, y: x.pow(3),
                lambda x, y: x.pow_(3),
                lambda x, y: x.pow(0.0),
                lambda x, y: x.pow(1.0),
                lambda x, y: x.reciprocal(),
                lambda x, y: x.remainder(2),
                lambda x, y: x.round(),
                lambda x, y: x.round_(),
                lambda x, y: x.rsqrt(),
                lambda x, y: x.rsqrt_(),
                lambda x, y: x.sigmoid(),
                lambda x, y: x.sigmoid_(),
                lambda x, y: x.logit(),
                lambda x, y: x.logit_(),
                lambda x, y: x.logit(1e-6),
                lambda x, y: x.logit_(1e-6),
                lambda x, y: x.sign(),
                lambda x, y: x.sign_(),
                lambda x, y: x.sgn(),
                lambda x, y: x.sgn_(),
                lambda x, y: x.sin(),
                lambda x, y: x.sin_(),
                lambda x, y: x.sinh(),
                lambda x, y: x.sinh_(),
                lambda x, y: x.sqrt(),
                lambda x, y: x.sqrt_(),
                lambda x, y: x.tan(),
                lambda x, y: x.tanh(),
                lambda x, y: x.trunc(),
                lambda x, y: x.trunc_(),
                _chunk_op,
                _unsqueeze_op_add,
                _unsqueeze_op_clone,
            ]
            for fn in fns:
                x_c = x.contiguous()
                y_c = y.contiguous()
                result_c = fn(x_c, y_c)
                result = fn(x, y)
                self.assertEqual(result, result_c)
                self.assertTrue(
                    torch.xpu.is_contiguous_channels_last_1d(result),
                    "result of the '{}' is not in '{}' format".format(inspect.getsource(fn).strip(), "channels last 1d"))

            for fn in bias_fns:
                x_c = x.contiguous()
                b_c = bias.contiguous()
                result_c = fn(x_c, b_c)
                result = fn(x, bias)
                self.assertEqual(result, result_c)
                self.assertTrue(
                    torch.xpu.is_contiguous_channels_last_1d(result),
                    "result of the '{}' is not in '{}' format".format(inspect.getsource(fn).strip(), "channels last 1d"))

            for fn in return_contig_fns:
                x_c = x.contiguous()
                y_c = y.contiguous()
                result_c = fn(x_c, y_c)
                result = fn(x, y)
                self.assertEqual(result, result_c)
                self.assertTrue(
                    result.is_contiguous(memory_format=torch.contiguous_format),
                    "result of the '{}' is not in '{}' format".format(inspect.getsource(fn).strip(), torch.contiguous_format))

        _test_helper(
            torch.xpu.to_channels_last_1d(torch.randn((4, 3, 8), device=dpcpp_device)),
            abs(torch.randn((4, 3, 8), device=dpcpp_device)) + 1,
            torch.xpu.to_channels_last_1d(torch.randn((1, 3, 1), device=dpcpp_device)))

    def _test_memory_format_transformations(self, input_generator_fn, transformation_fn,
                                            compare_data=True, default_is_preserve=False):

        # xc is a channels last 1d tensor
        xc = input_generator_fn()
        # xc is not memory dense, but looks like channels last 1d
        xc = xc[..., ::2]

        xc = input_generator_fn()
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        self.assertFalse(torch.xpu.is_contiguous_channels_last_1d(clone))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn()
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(torch.xpu.is_contiguous_channels_last_1d(clone))
        else:
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(torch.xpu.is_contiguous_channels_last_1d(clone))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=dpcpp_device)
        for _ in range(10):
            permutation = list(range(len(x.shape)))
            random.shuffle(permutation)
            x = x.permute(permutation)
            self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    def test_memory_format_to(self):
        def get_generator(shape):
            def input_generator_fn():
                return torch.xpu.to_channels_last_1d(torch.randn(shape, device=dpcpp_device, dtype=torch.float32))
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.to(dtype=torch.float64, **kwargs)

        shape = (4, 3, 8)
        self._test_memory_format_transformations(
            get_generator(shape), transformation_fn, default_is_preserve=True)

    def test_memory_format_type(self):
        def get_generator(shape):
            def input_generator_fn():
                return torch.xpu.to_channels_last_1d(torch.randn(shape, device=dpcpp_device, dtype=torch.float32))
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.to(torch.float64, **kwargs)

        shape = (4, 3, 8)
        self._test_memory_format_transformations(
            get_generator(shape), transformation_fn, default_is_preserve=True)

    def test_memory_format_clone(self):
        def get_generator(shape):
            def input_generator_fn():
                return torch.xpu.to_channels_last_1d(torch.randn(shape, device=dpcpp_device, dtype=torch.float32))
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.clone(**kwargs)

        shape = (4, 3, 8)
        self._test_memory_format_transformations(
            get_generator(shape), transformation_fn, default_is_preserve=True)

    def test_memory_format_type_shortcuts(self):
        def get_generator(shape, dtype):
            def input_generator_fn():
                return torch.xpu.to_channels_last_1d(torch.randn(shape, device=dpcpp_device, dtype=dtype).clamp(0, 1).round())
            return input_generator_fn


        def get_fn(fn_name):
            def transformation_fn(tensor, **kwargs):
                fn = getattr(tensor, fn_name)
                return fn(**kwargs)
            return transformation_fn

        shortcuts = ['byte', 'char', 'double', 'bool', 'half', 'int', 'long', 'short']

        shape = (4, 3, 8)
        for fn_name in shortcuts:
            self._test_memory_format_transformations(
                get_generator(shape, torch.float32), get_fn(fn_name), default_is_preserve=True)

        # Test 'float' separately to avoid float->float no-op.
        self._test_memory_format_transformations(
            get_generator(shape, torch.float64), get_fn('float'), default_is_preserve=True)
