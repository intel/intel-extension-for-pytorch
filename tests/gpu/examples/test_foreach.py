from numbers import Number
import random
import re

from torch.testing._internal.common_utils import run_tests, TEST_WITH_SLOW, TestCase
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipMeta, ops)
from torch.testing._internal.common_methods_invocations import \
    (foreach_unary_op_db)
from torch.testing._internal.common_dtype import (
    get_all_dtypes, get_all_complex_dtypes,
)

N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]
Scalars = (
    random.randint(1, 10),
    1.0 - random.random(),
    True,
    complex(1.0 - random.random(), 1.0 - random.random()),
)


def getScalarLists(N):
    return (
        ("int", [random.randint(0, 9) + 1 for _ in range(N)]),
        ("float", [1.0 - random.random() for _ in range(N)]),
        ("complex", [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)]),
        ("bool", [True for _ in range(N)]),
        ("mixed", [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 3)]),
        ("mixed", [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 4)]),
    )


_BOOL_SUB_ERR_MSG = "Subtraction, the `-` operator"


class RegularFuncWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, values=None, **kwargs):
        if values is not None:
            assert len(inputs) == 3
            if isinstance(values, Number):
                values = [values for _ in range(len(inputs[0]))]
            return [self.func(*i, value=values[idx], **kwargs) for idx, i in enumerate(zip(*inputs))]
        if len(inputs) == 2 and isinstance(inputs[1], Number):
            # binary op with tensorlist and scalar.
            inputs[1] = [inputs[1] for _ in range(len(inputs[0]))]
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class ForeachFuncWrapper:

    def __init__(self, func, n_expected_xpuLaunchKernels):
        self.func = func
        self.n_expected_xpuLaunchKernels = n_expected_xpuLaunchKernels
        # Some foreach functions don't have in-place implementations.
        self._is_inplace = False if func is None else func.__name__.endswith('_')

    def __call__(self, inputs, is_xpu, is_fastpath, **kwargs):
        actual = None
        actual = self.func(*inputs, **kwargs)
        # note(mkozuki): inplace foreach functions are void functions.
        return inputs[0] if self._is_inplace else actual


class TestForeach(TestCase):
    @property
    def is_xpu(self):
        return self.device_type == 'xpu'

    # note(mkozuki): It might be the case that the expected number of `xpuLaunchKernel`s
    # is greater than 1 once foreach functions internally separate their input `TensorList`s by
    # devices & dtypes into vectors of tensors.
    def _get_funcs(self, op, n_expected_xpuLaunchKernels):
        return (
            ForeachFuncWrapper(op.method_variant, n_expected_xpuLaunchKernels),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant, n_expected_xpuLaunchKernels),
            RegularFuncWrapper(op.ref_inplace),
        )

    def _regular_unary_test(self, dtype, op, ref, inputs, is_fastpath):
        if is_fastpath:
            self.assertEqual(ref(inputs), op(inputs, self.is_xpu, is_fastpath))
            return
        try:
            actual = op(inputs, self.is_xpu, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(inputs)
        else:
            expected = ref(inputs)
            self.assertEqual(actual, expected)

    # note(mkozuki): why `try-except` for both fastpath?
    # - inputs for fastpath can be integer tensors.
    #    - this is becase opinfo dtypes are configured for outpulace implementation
    # - for integer inputs, trigonometric functions and exponential function returns float outputs,
    #   which causes "result type Float can't be case to the desired type" error.
    # Thus, `try-except` is used even if `is_fastpath` is `True`.
    def _inplace_unary_test(self, dtype, inplace, inplace_ref, inputs, is_fastpath):
        copied_inputs = [[t.clone().detach() for t in tensors] for tensors in inputs]
        try:
            inplace(inputs, self.is_xpu, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                inplace_ref(copied_inputs)
        else:
            inplace_ref(copied_inputs),
            self.assertEqual(copied_inputs, inputs)

    def _test_unary(self, device, dtype, opinfo, N, is_fastpath):
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, 1)
        inputs = opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        # note(mkozuki): Complex inputs for `_foreach_abs` go through slowpath.
        if opinfo.name == "_foreach_abs" and dtype in get_all_complex_dtypes():
            is_fastpath = False
        self._regular_unary_test(dtype, op, ref, inputs, is_fastpath)
        self._inplace_unary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath)

    @skipMeta
    @ops(foreach_unary_op_db)
    def test_unary_fastpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=True)

    @dtypes(*get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=False)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
