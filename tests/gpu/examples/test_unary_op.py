from functools import partial, update_wrapper

import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import ipex


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


class TetsTorchMethod(TestCase):
    @repeat_test_for_types(FLOATING_DTYPES)
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

    @repeat_test_for_types(INTEGRAL_DTYPES)
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
