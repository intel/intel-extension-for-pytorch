from functools import partial, update_wrapper

import torch
from torch.testing._internal.common_utils import (TestCase)

from torch.testing._internal.common_device_type import (dtypes)

import intel_extension_for_pytorch # noqa


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

FLOATING_DTYPES = [
    torch.float,
    torch.double,
    # torch.half,
    # torch.bfloat16,
]

INTEGRAL_DTYPES = [
    # torch.short,
    torch.int,
    torch.long,
    # torch.uint8,
    torch.int8,
    # torch.bool,
]

'''
COMPLEX_DTYPES=[
    torch.cfloat,
    torch.cdouble
    ]
'''

OP_TEST_FOR_FLOATING = [
    torch.neg,
    torch.cos,
    torch.sin,
    torch.tan,
    torch.cosh,
    torch.sinh,
    torch.tanh,
    torch.acos,
    torch.asin,
    torch.atan,
    torch.ceil,
    torch.expm1,
    torch.round,
    torch.frac,
    torch.trunc,
    wrapped_partial(torch.clamp, min=-0.1, max=0.5),
    torch.erf,
    torch.erfc,
    torch.exp,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logit,
    torch.rsqrt,
    torch.sqrt,
    torch.erfinv,
    torch.digamma,
    torch.sign,
    torch.reciprocal,
    torch.conj,
]

OP_TEST_FOR_FLOATING_ = [
    "cos_()",
]

OP_TEST_FOR_INTEGER = [
    torch.bitwise_not,
    torch.logical_not,
]

OP_TEST_FOR_INTEGER_ = [
    "__and__(3)",
    "__iand__(3)",
    "__or__(3)",
    "__ior__(3)",
]

OP_TEST_FOR_BACKWARD = [
    torch.logit
]


class TetsTorchMethod(TestCase):
    @dtypes(FLOATING_DTYPES)
    def test_unary_op_for_floating(self, dtype=torch.float):
        a = torch.randn(
            [2, 2, 2, 2], device=cpu_device, dtype=torch.float)

        x_cpu = torch.as_tensor(a, dtype=dtype, device=cpu_device)
        x_xpu = torch.as_tensor(a, dtype=dtype, device=xpu_device)

        for op in OP_TEST_FOR_FLOATING:
            y_cpu = op(x_cpu)
            y_xpu = op(x_xpu)
            print("CPU {}: {}".format(op.__name__, y_cpu))
            print("XPU {}: {}".format(op.__name__, y_xpu.cpu()))
            self.assertEqual(y_cpu, y_xpu.cpu())

        for op_str in OP_TEST_FOR_FLOATING_:
            exec("""
x_cpu_clone = x_cpu.clone()
x_xpu_clone = x_xpu.clone()
y_cpu = x_cpu_clone.{}
y_xpu = x_xpu_clone.{}
print("CPU {}: ", y_cpu)
print("XPU {}: ", y_xpu.cpu())
self.assertEqual(y_cpu, y_xpu.cpu())
                """.format(op_str, op_str, op_str, op_str)
                 )

    @dtypes(INTEGRAL_DTYPES)
    def test_unary_op_for_integer(self, dtype=torch.int):
        a = torch.randint(
            -5, 5, [2, 2, 2, 2], device=cpu_device, dtype=torch.int)

        x_cpu = torch.as_tensor(a, dtype=dtype, device=cpu_device)
        x_xpu = torch.as_tensor(a, dtype=dtype, device=xpu_device)

        for op in OP_TEST_FOR_INTEGER:
            y_cpu = op(x_cpu)
            y_xpu = op(x_xpu)
            print("CPU {}: {}".format(op.__name__, y_cpu))
            print("XPU {}: {}".format(op.__name__, y_xpu.cpu()))
            self.assertEqual(y_cpu, y_xpu.cpu())

        for op_str in OP_TEST_FOR_INTEGER_:
            exec("""
x_cpu_clone = x_cpu.clone()
x_xpu_clone = x_xpu.clone()
y_cpu = x_cpu_clone.{}
y_xpu = x_xpu_clone.{}
print("CPU {}: ", y_cpu)
print("XPU {}: ", y_xpu.cpu())
self.assertEqual(y_cpu, y_xpu.cpu())
                """.format(op_str, op_str, op_str, op_str)
                 )

    @dtypes([torch.float, torch.half, torch.bfloat16])
    def test_unary_op_signbit(self, dtype=torch.float):
        x_cpu = torch.randn(5, 5, requires_grad=True)
        sign_cpu = torch.signbit(x_cpu)
        x_xpu = x_cpu.clone().to(xpu_device)
        sign_xpu = torch.signbit(x_xpu)
        self.assertEqual(sign_cpu, sign_xpu.to(cpu_device))
        sign_xpu = torch.signbit(sign_xpu)
        self.assertFalse(sign_xpu.any())

    def _test_unary_backward(self, op, x_cpu, x_xpu, param):
        if (len(param) == 0):
            y_cpu = op(x_cpu)
            y_xpu = op(x_xpu)
        else:
            y_cpu = op(x_cpu, **param)
            y_xpu = op(x_xpu, **param)

        y_cpu.backward(x_cpu)
        y_xpu.backward(x_xpu)
        print("CPU {}: {}".format(op.__name__, x_cpu.grad))
        print("XPU {}: {}".format(op.__name__, x_xpu.grad))
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())

    @dtypes(FLOATING_DTYPES)
    def test_unary_backward_for_floating(self, dtype=torch.float):
        x_cpu = torch.randn(
            [2, 2, 2, 2], device=cpu_device, dtype=dtype, requires_grad=True)
        x_xpu = x_cpu.clone().detach().to("xpu").requires_grad_(True)


        for op in OP_TEST_FOR_BACKWARD:
            param = {}
            if op == torch.logit:
                param['eps'] = 0.15
            self._test_unary_backward(op, x_cpu, x_xpu, param)
