from collections import defaultdict
from torch import Tensor
import torch.autograd
from torch._decomp import decomposition_table
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils._mode_utils import no_dispatch
from torch.testing._internal.common_utils import is_iterable_of_tensors, TestCase, skipIfCrossRef, suppress_warnings, TEST_WITH_ASAN, run_tests, skipIfSlowGradcheckEnv, skipIfTorchDynamo
from torch.testing._internal.common_device_type import onlyNativeDeviceTypes, ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
from torch._dispatch.python import enable_python_dispatcher
import itertools
import functools
from functools import partial
import unittest
aten = torch.ops.aten
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest

def overload_to_aten_name(overload):
    return overload._schema.name.split('::')[1]
decomposition_names = {overload_to_aten_name(k) for k in decomposition_table}
_decomp_test_ops = [op for op in op_db if op.aten_name in decomposition_names or op.aten_backward_name in decomposition_names]

def diff_arg(arg, requires_grad=True):

    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all([is_differentiable_arg(a) for a in arg]):
            return True
        if all([not is_differentiable_arg(a) for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)

def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    (inputs, inputs_spec) = tree_flatten(inputs)
    diff_inputs = tuple((inp for inp in inputs if inp.requires_grad))
    if grad_outputs is None:
        diff_outputs = tuple((out for out in outputs if out.requires_grad))
    else:
        diff_grad_outputs = [(out, go) for (out, go) in zip(outputs, grad_outputs) if out.requires_grad]
        if len(diff_grad_outputs) == 0:
            (diff_outputs, grad_outputs) = ((), ())
        else:
            (diff_outputs, grad_outputs) = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(diff_outputs, diff_inputs, grad_outputs, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)

def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)

def ref_vjp_no_create(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents), create_graph=False)
    return (result, wrapped)
dtype_precisions = {torch.float16: (0.001, 1e-05), torch.bfloat16: (0.016, 0.0001), torch.float32: (1.3e-06, 1e-05), torch.float64: (1e-07, 1e-07), torch.complex32: (0.001, 1e-05), torch.complex64: (1.3e-06, 1e-05), torch.complex128: (1e-07, 1e-07)}

def _getDefaultRtolAndAtol(dtype0, dtype1):
    rtol = max(dtype_precisions.get(dtype0, (0, 0))[0], dtype_precisions.get(dtype1, (0, 0))[0])
    atol = max(dtype_precisions.get(dtype0, (0, 0))[1], dtype_precisions.get(dtype1, (0, 0))[1])
    return (rtol, atol)

def op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    assert orig.dtype == decomp.dtype, f'{i} Operation:  {op}'
    if orig.numel() == 0 or decomp.numel() == 0:
        assert orig.numel() == decomp.numel()
        return
    assert orig.shape == decomp.shape, f'{i} Operation:  {op}'
    tol_table = {(torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 0.001, (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 0.02, (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-05, (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-05, (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-06, (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-06}
    if ref.is_floating_point():
        orig_diff = (orig - ref).abs().max()
        decomp_diff = (decomp - ref).abs().max()
        atol = tol_table.get((test_dtype, op), 1e-07)
        if decomp_diff > orig_diff + atol:
            raise RuntimeError(f'Difference from float64 is larger with decomposition {op.__name__} than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\natol = {atol}\nargs = {args}\nkwargs = {kwargs}')
    else:
        test_case.assertEqual(orig, decomp, msg=f'{op.__name__}\nargs = {args}\nkwargs = {kwargs}')

def op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    test_case.assertEqual(orig.dtype, decomp.dtype, f'Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}')
    tol_table = {(torch.float32, torch.ops.aten.native_layer_norm.default): (0.001, 0.001), (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (0.001, 0.001), (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-06, 1e-06), (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-06, 3e-05), (torch.float32, torch.ops.aten.mv.default): (1e-05, 3e-05), (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-05, 1e-06)}
    if (test_dtype, op) in tol_table:
        (rtol, atol) = tol_table[decomp.dtype, op]
    else:
        (rtol, atol) = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    test_case.assertEqual(orig, decomp, rtol=rtol, atol=atol, msg=f'{op.__name__}\nargs = {args}\nkwargs = {kwargs}')

def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    (flat_args, args_spec) = tree_flatten(args)
    diff_argnums = tuple((i for (i, arg) in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad)))
    assert len(diff_argnums) > 0
    primals = tuple((flat_args[i] for i in diff_argnums))

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for (num, arg) in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple(result)
            result = tuple((r for r in result if isinstance(r, Tensor) and (r.is_floating_point() or r.is_complex())))
            assert len(result) > 0
        return result
    return (wrapped, primals)

def upcast_tensor(x, dtype=torch.float32):
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        return x.to(dtype=dtype)
    elif isinstance(x, torch.dtype) and x in [torch.float16, torch.bfloat16, torch.float]:
        return dtype
    else:
        return x

def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad)
CROSS_REF_EXCLUDE_SET = {('xpu', torch.bfloat16, 'nn.functional.bilinear'), ('xpu', torch.float16, 'nn.functional.dropout'), ('xpu', torch.bfloat16, 'nn.functional.dropout'), ('xpu', torch.float64, 'nn.functional.dropout'), ('xpu', torch.float32, 'nn.functional.dropout'), (None, None, 'new_empty'), (None, None, 'empty_like'), (None, None, 'empty'), (None, None, 'nn.functional.relu6'), (None, None, 'meshgrid')}
CROSS_REF_BACKWARD_EXCLUDE_SET = {('xpu', torch.bfloat16, 'nn.functional.embedding')}
all_decomposed = set()
all_called = defaultdict(int)
'\nimport atexit\ndef check_coverage():\n    print("missing coverage:")\n    print("\n".join(map(str, decomposition_table.keys() - all_decomposed)))\natexit.register(check_coverage)\n'
"\nimport atexit\ndef dump_ops():\n    with open('run_ops.txt', 'w') as f, open('count_ops.txt', 'w') as g:\n        for op, count in sorted(all_called.items(), key=lambda x: x[0].__name__):\n            f.write(f'{op.__name__}\n')\n            g.write(f'{count}\n')\n    with open('run_decompositions.txt', 'w') as f:\n        for op in sorted([i.__name__ for i in all_decomposed]):\n            f.write(f'{op}\n')\n\natexit.register(dump_ops)\n"

def any_unsupported(args, kwargs):

    def test_unsupported(t):
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            return any([t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized, t.is_nested, torch._is_functional_tensor(t)])
        elif torch.overrides.is_tensor_like(t):
            return True
        else:
            return False
    (flat_args, _) = tree_flatten(args)
    (flat_kwargs, _) = tree_flatten(kwargs)
    return any((test_unsupported(x) for x in itertools.chain(flat_args, flat_kwargs)))

@skipIfSlowGradcheckEnv
class TestDecomp(TestCase):
    longMessage = True

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    def test_quick(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=False)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=True)

    @skipIfTorchDynamo('Test does not work with TorchDynamo')
    def do_cross_ref(self, device, dtype, op, *, run_all):
        test_keys = [(torch.device(device).type, dtype, op.name), (None, dtype, op.name), (None, None, op.name)]
        if any((key in CROSS_REF_EXCLUDE_SET for key in test_keys)):
            self.skipTest(f'{op.name} in {dtype} not supported')
        skip_decomp_vjp = any((key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys))
        test_dtype = dtype
        called = set()
        decomposed = set()
        saved_precision = self.precision
        saved_rel_tol = self.rel_tol
        test_case = self

        class DecompCrossRefMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                with no_dispatch():
                    return self._torch_dispatch(func, types, args, kwargs)

            def _torch_dispatch(self, func, types, args=(), kwargs=None):
                test_case.precision = saved_precision
                test_case.rel_tol = saved_rel_tol
                called.add(func)
                all_called[func] += 1
                if func not in decomposition_table or func in [torch.ops.aten.detach.default, torch.ops.aten.empty.memory_format, torch.ops.aten.empty_like.default, torch.ops.aten.new_empty.default] or any_unsupported(args, kwargs):
                    return func(*args, **kwargs)
                decomposed.add(func)
                all_decomposed.add(func)
                decomposition = decomposition_table[func]
                do_relative_check = test_dtype in [torch.float16, torch.bfloat16]
                real_out_unflat = func(*args, **kwargs)
                (real_out, _) = tree_flatten(real_out_unflat)
                (decomp_out, _) = tree_flatten(decomposition(*args, **kwargs))
                assert len(real_out) == len(decomp_out)
                if do_relative_check:
                    upcast = partial(upcast_tensor, dtype=torch.float64)
                    (real_out_double, _) = tree_flatten(func(*tree_map(upcast, args), **tree_map(upcast, kwargs)))
                    for (i, orig, decomp, ref) in zip(range(len(real_out)), real_out, decomp_out, real_out_double):
                        if not isinstance(orig, torch.Tensor):
                            assert type(orig) == type(decomp)
                            assert orig == decomp
                            continue
                        op_assert_ref(test_case, func, test_dtype, i, orig, decomp, ref, args, kwargs)
                else:
                    for (orig, decomp) in zip(real_out, decomp_out):
                        if not isinstance(orig, torch.Tensor):
                            assert type(orig) == type(decomp)
                            assert orig == decomp
                            continue
                        op_assert_equal(test_case, func, test_dtype, orig, decomp, args, kwargs)
                return real_out_unflat
        requires_grad = op.supports_autograd and dtype in op.supported_backward_dtypes(torch.device(device).type) and (not dtype == torch.complex32)
        samples = op.sample_inputs(device, test_dtype, requires_grad=requires_grad)

        def check_decomposed(aten_name):
            self.assertTrue(any((overload_to_aten_name(c) == aten_name for c in decomposed)), msg=f"aten.{aten_name} was not decomposed, saw calls for: {', '.join(map(str, list(called)))}. If your op is  CompositeImplicitAutograd you should skip this test by updating CROSS_REF_EXCLUDE_SET.")
        aten_name = op.decomp_aten_name or op.aten_name
        func = op.get_op()
        for sample_input in samples:
            if requires_grad:
                (fn, primals) = normalize_op_input_output(func, sample_input)
                primals = tree_map(lambda x: x if isinstance(x, torch.Tensor) else x, primals)
                decomposed.clear()
                with DecompCrossRefMode(), enable_python_dispatcher():
                    (decomp_out, decomp_vjp_fn) = ref_vjp_no_create(fn, *primals)
                if aten_name in decomposition_names:
                    check_decomposed(aten_name)
                if not skip_decomp_vjp and (op.aten_backward_name in decomposition_names or run_all):
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)
                    decomposed.clear()
                    with DecompCrossRefMode(), enable_python_dispatcher():
                        decomp_vjp_fn(cotangents)
                    if not run_all:
                        check_decomposed(op.aten_backward_name)
            elif aten_name in decomposition_names or run_all:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                decomposed.clear()
                with DecompCrossRefMode(), enable_python_dispatcher():
                    func(*args, **kwargs)
                if not run_all:
                    check_decomposed(aten_name)
            else:
                assert op.supports_autograd
                self.skipTest("only backwards is decomposed, but dtype doesn't support AD")
instantiate_device_type_tests(TestDecomp, globals())

class DecompContiguousTests(TestCase):

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_softmax(self, device):
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)
        ref = torch.ops.aten._softmax(x, -1, False)
        res = torch._decomp.decompositions._softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_log_softmax(self, device):
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)
        ref = torch.ops.aten._log_softmax(x, -1, False)
        res = torch._decomp.decompositions._log_softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())
instantiate_device_type_tests(DecompContiguousTests, globals())
if __name__ == '__main__':
    run_tests()