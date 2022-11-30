from collections.abc import Sequence
from functools import partial
import warnings
import unittest
import itertools
import torch
import contextlib
from collections import defaultdict
from importlib import import_module
from torch.utils._pytree import tree_map
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import floating_and_complex_types_and, all_types_and_complex_and
from test_proxy_tensor import xfail, skip, skipOps
from torch.testing._internal.common_utils import TestCase, is_iterable_of_tensors, run_tests, IS_SANDCASTLE, clone_input_helper, IS_CI, suppress_warnings, noncontiguous_like, TEST_WITH_ASAN, TEST_WITH_UBSAN, skipIfRocm, IS_WINDOWS, IS_FBCODE, first_sample, parametrize, skipIfSlowGradcheckEnv
from torch.testing._internal.common_methods_invocations import op_db, UnaryUfuncInfo, ReductionOpInfo, ReductionPythonRefInfo, SpectralFuncInfo, ops_and_refs, python_ref_db, BinaryUfuncInfo
from torch.testing._internal.common_device_type import deviceCountAtLeast, instantiate_device_type_tests, ops, onlyCUDA, onlyCPU, onlyNativeDeviceTypes, OpDTypes, skipCUDAIfRocm, skipMeta
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.fake_utils import outputs_alias_inputs
import torch._prims as prims
from torch._prims.context import TorchRefsMode
from torch.testing._internal import opinfo
from torch.testing._internal import composite_compliance
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode
torch.set_default_dtype(torch.float32)
_variant_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat))
_ref_test_ops = tuple(filter(lambda op: not isinstance(op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)) and op.ref is not None, op_db))
_ops_and_refs = op_db + python_ref_db
aten = torch.ops.aten
from common.pytorch_test_base import TestCase, dtypesIfXPU, TEST_XPU, TEST_MULTIGPU, largeTensorTest

@skipIfSlowGradcheckEnv
class TestCommon(TestCase):
    exact_dtype = True

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if IS_CI:
            err_msg = 'The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries.This is OK for testing, but be sure to set the dtypes manually before landing your PR!'
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += '\n' + fmt_str
            assert len(filtered_ops) == 0, err_msg

    @onlyCUDA
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for xpu_device_str in devices:
            xpu_device = torch.device(xpu_device_str)
            samples = op.sample_inputs(xpu_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)
            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == xpu_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(map(lambda t: t.device == xpu_device, result)))
            else:
                self.skipTest('Skipped! Only supports single tensor or iterable of tensor outputs.')

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_numpy_ref(self, device, dtype, op):
        try:
            cur_default = torch.get_default_dtype()
            torch.set_default_dtype(torch.double)
            for sample_input in op.reference_inputs(device, dtype):
                self.compare_with_reference(op, op.ref, sample_input, exact_dtype=dtype is not torch.long)
        finally:
            torch.set_default_dtype(cur_default)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @ops(python_ref_db)
    def test_python_ref_meta(self, device, dtype, op):
        with FakeTensorMode() as mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                out = FakeTensor.from_tensor(x, mode)
                return out
            return x
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            result = op(sample.input, *sample.args, **sample.kwargs)
            meta_sample = sample.transform(_to_tensormeta)
            try:
                with mode:
                    meta_result = op(meta_sample.input, *meta_sample.args, **meta_sample.kwargs)
            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                continue
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                continue
            if isinstance(result, torch.Tensor):
                self.assertTrue(isinstance(meta_result, FakeTensor))
                prims.utils.compare_tensor_meta(result, meta_result)
            elif isinstance(result, Sequence):
                for (a, b) in zip(result, meta_result):
                    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                        self.assertTrue(isinstance(b, FakeTensor))
                        prims.utils.compare_tensor_meta(a, b)

    def _ref_test_helper(self, ctx, device, dtype, op, skip_zero_numel=False, skip_zero_dim=False, skip_bfloat=False, skip_view_consistency=False):
        ex = None
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            if isinstance(sample.input, torch.Tensor) and sample.input.numel() == 0 and skip_zero_numel:
                continue
            if isinstance(sample.input, torch.Tensor) and sample.input.ndim == 0 and skip_zero_dim:
                continue
            is_lower_than_xpu11_0 = torch.version.xpu is not None and [int(x) for x in torch.version.xpu.split('.')] < [11, 0]
            if skip_bfloat and is_lower_than_xpu11_0 and (isinstance(sample.input, torch.Tensor) and sample.input.dtype == torch.bfloat16 or any((isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16 for arg in sample.args))):
                continue
            with ctx():
                ref_result = op(sample.input, *sample.args, **sample.kwargs)
            torch_result = op.torch_opinfo(sample.input, *sample.args, **sample.kwargs)
            for (a, b) in zip(tree_flatten(ref_result)[0], tree_flatten(torch_result)[0]):
                if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                    prims.utils.compare_tensor_meta(a, b)
                    if getattr(op, 'validate_view_consistency', True) and (not skip_view_consistency):
                        msg = f"The torch implementation {('returns' if b._is_view() else 'does not return')} a view, while the reference {('does' if a._is_view() else 'does not')}"
                        self.assertEqual(a._is_view(), b._is_view(), msg)
            precise_dtype = torch.bool
            if prims.utils.is_integer_dtype(dtype):
                precise_dtype = dtype
            if prims.utils.is_float_dtype(dtype):
                precise_dtype = torch.double
            if prims.utils.is_complex_dtype(dtype):
                precise_dtype = torch.cdouble
            try:
                self.assertEqual(ref_result, torch_result, exact_stride=False, exact_device=True, exact_layout=True, exact_is_coalesced=True)
            except AssertionError as e:
                if dtype is precise_dtype:
                    raise e
                ex = e
            if not ex:
                continue

            def _make_precise(x):
                if isinstance(x, torch.dtype):
                    return precise_dtype
                if isinstance(x, torch.Tensor) and x.dtype is dtype:
                    return x.to(precise_dtype)
                return x
            precise_sample = sample.transform(_make_precise)
            precise_result = op.torch_opinfo(precise_sample.input, *precise_sample.args, **precise_sample.kwargs)

            def _distance(a, b):
                if prims.utils.is_boolean_dtype(a.dtype):
                    assert b.dtype is torch.bool
                    return (a ^ b).sum()
                same = a == b
                if prims.utils.is_float_dtype(a.dtype) or prims.utils.is_complex_dtype(a.dtype):
                    same = torch.logical_or(same, torch.logical_and(torch.isnan(a), torch.isnan(b)))
                actual_error = torch.where(same, 0, torch.abs(a - b)).sum()
                return actual_error
            ref_distance = 0
            for (a, b) in zip(tree_flatten(ref_result)[0], tree_flatten(precise_result)[0]):
                ref_distance = ref_distance + _distance(a, b)
            torch_distance = 0
            for (a, b) in zip(tree_flatten(torch_result)[0], tree_flatten(precise_result)[0]):
                torch_distance = torch_distance + _distance(a, b)
            msg = f'Reference result was farther ({ref_distance}) from the precise computation than the torch result was ({torch_distance})!'
            self.assertTrue(ref_distance <= torch_distance, msg=msg)
        if ex is not None:
            msg = 'Test passed because the reference was more accurate than the torch operator.'
            warnings.warn(msg)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @ops(python_ref_db)
    def test_python_ref(self, device, dtype, op):
        self._ref_test_helper(lambda : TorchRefsMode(strict=True), device, dtype, op)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @ops(python_ref_db)
    def test_python_ref_torch_fallback(self, device, dtype, op):
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyCUDA
    @skipCUDAIfRocm
    @ops(python_ref_db)
    @parametrize('executor', ['aten', 'nvfuser'])
    def test_python_ref_executor(self, device, dtype, op, executor):
        from torch._prims_common import _torch_dtype_to_nvfuser_dtype_map
        if executor == 'nvfuser' and dtype not in _torch_dtype_to_nvfuser_dtype_map:
            raise unittest.SkipTest(f"nvfuser doesn't support dtype {dtype}")
        if executor == 'nvfuser' and dtype not in [torch.int32, torch.float32]:
            raise unittest.SkipTest('skipped for speed')
        if executor == 'nvfuser' and (not op.supports_nvfuser):
            raise unittest.SkipTest(f"{op.name} doesn't support nvfuser")
        skip_zero_dim = False
        if executor == 'nvfuser' and isinstance(op, ReductionPythonRefInfo):
            skip_zero_dim = True
        normalization_ops = ['_refs.softmax', '_refs.logsumexp', '_refs.log_softmax', '_refs.sum_to_size']
        if executor == 'nvfuser' and op.name in normalization_ops:
            skip_zero_dim = True
        from torch._prims.executor import make_traced
        from copy import copy
        op = copy(op)
        executor = 'strictly_nvfuser' if executor == 'nvfuser' else executor
        op.op = partial(make_traced(op.op), executor=executor)
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op, skip_zero_numel='nvfuser' in executor, skip_zero_dim=skip_zero_dim, skip_bfloat='nvfuser' in executor, skip_view_consistency='nvfuser' in executor)

    @skipMeta
    @onlyNativeDeviceTypes
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                op(si.input, *si.args, **si.kwargs)

    @skipMeta
    @onlyNativeDeviceTypes
    @ops([op for op in python_ref_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_python_ref_errors(self, device, op):
        mode = FakeTensorMode()
        with mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                return FakeTensor.from_tensor(x, mode)
            return x
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            meta_sample = si.transform(_to_tensormeta)
            with self.assertRaisesRegex(ei.error_type, ''):
                op(meta_sample.input, *meta_sample.args, **meta_sample.kwargs)

    @unittest.skipIf(IS_WINDOWS, 'Skipped under Windows')
    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    def test_noncontiguous_samples(self, device, dtype, op):
        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)
        for sample_input in sample_inputs:
            (t_inp, t_args, t_kwargs) = (sample_input.input, sample_input.args, sample_input.kwargs)
            noncontig_sample = sample_input.noncontiguous()
            (n_inp, n_args, n_kwargs) = (noncontig_sample.input, noncontig_sample.args, noncontig_sample.kwargs)
            sample_tensor = t_inp if isinstance(t_inp, torch.Tensor) else t_inp[0]
            assert sample_tensor.grad is None
            assert sample_tensor.grad_fn is None
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)
            self.assertEqual(actual, expected)
            if not test_grad:
                continue
            expected = sample_input.output_process_fn_grad(expected)
            actual = sample_input.output_process_fn_grad(actual)
            if isinstance(expected, torch.Tensor):
                grad_for_expected = torch.randn_like(expected)
                grad_for_actual = noncontiguous_like(grad_for_expected)
            elif isinstance(expected, Sequence):
                expected = [t for t in expected if isinstance(t, torch.Tensor) and t.requires_grad]
                actual = [n for n in actual if isinstance(n, torch.Tensor) and n.requires_grad]
                grad_for_expected = [torch.randn_like(t) for t in expected]
                grad_for_actual = [noncontiguous_like(n) for n in grad_for_expected]
            else:
                continue
            t_inputs = (t_inp,) + t_args if isinstance(t_inp, torch.Tensor) else tuple(t_inp) + t_args
            n_inputs = (n_inp,) + n_args if isinstance(n_inp, torch.Tensor) else tuple(n_inp) + n_args
            t_input_tensors = [t for t in t_inputs if isinstance(t, torch.Tensor) and t.requires_grad]
            n_input_tensors = [n for n in n_inputs if isinstance(n, torch.Tensor) and n.requires_grad]
            self.assertEqual(len(t_input_tensors), len(n_input_tensors))
            t_grads = torch.autograd.grad(expected, t_input_tensors, grad_for_expected, allow_unused=True)
            n_grads = torch.autograd.grad(actual, n_input_tensors, grad_for_actual, allow_unused=True)
            msg = 'Got different gradients for contiguous / non-contiguous inputs wrt input {}.'
            for (i, (t, n)) in enumerate(zip(t_grads, n_grads)):
                self.assertEqual(t, n, msg=msg.format(i))

    @ops(_ops_and_refs, dtypes=OpDTypes.none)
    def test_out_warning(self, device, op):
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest('Skipped! Op has not supported dtypes on this device.')
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]
        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)
            if not isinstance(expected, torch.Tensor) and (not is_iterable_of_tensors(expected, include_empty=True)):
                self.skipTest('Skipped! Only supports single tensor or iterable of tensor outputs.')
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)
                return tuple(map(fn, out))

            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)
                return tuple(map(lambda t: t.stride(), out))

            def _extract_data_ptrs(out):
                if self.device_type != 'cpu' and self.device_type != 'xpu':
                    return ()
                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)
                return tuple(map(lambda t: t.data_ptr(), out))

            @suppress_warnings
            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)
                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)
                self.assertEqual(expected, out)
                if compare_strides_and_data_ptrs:
                    stride_msg = 'Strides are not the same! Original strides were {0} and strides are now {1}'.format(original_strides, final_strides)
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            def _case_zero_transform(t):
                wrong_shape = list(t.shape)
                if len(wrong_shape) == 0:
                    wrong_shape = [2]
                else:
                    wrong_shape[-1] = wrong_shape[-1] + 1
                return make_tensor(wrong_shape, dtype=t.dtype, device=t.device)
            _compare_out(_case_zero_transform, compare_strides_and_data_ptrs=False)

            def _any_nonempty(out):
                if isinstance(out, torch.Tensor):
                    return out.numel() > 0
                return any((x.numel() > 0 for x in out))
            out = _apply_out_transform(_case_zero_transform, expected)
            msg_fail = 'Resized a non-empty tensor but did not warn about it.'
            if _any_nonempty(out):
                with self.assertWarnsRegex(UserWarning, 'An output with one or more elements', msg=msg_fail):
                    op_out(out=out)

    @ops(_ops_and_refs, dtypes=OpDTypes.any_one)
    def test_out(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)
            if not isinstance(expected, torch.Tensor) and (not is_iterable_of_tensors(expected, include_empty=True)):
                self.skipTest('Skipped! Only supports single tensor or iterable of tensor outputs.')
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)
                return tuple(map(fn, out))

            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)
                return tuple(map(lambda t: t.stride(), out))

            def _extract_data_ptrs(out):
                if self.device_type != 'cpu' and self.device_type != 'xpu':
                    return ()
                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)
                return tuple(map(lambda t: t.data_ptr(), out))

            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)
                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)
                self.assertEqual(expected, out)
                if compare_strides_and_data_ptrs:
                    stride_msg = 'Strides are not the same! Original strides were {0} and strides are now {1}'.format(original_strides, final_strides)
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            def _case_zero_transform(t):
                try:
                    info = torch.iinfo(t.dtype)
                    return torch.full_like(t, info.max)
                except TypeError as te:
                    return torch.full_like(t, float('nan'))
            _compare_out(_case_zero_transform)

            def _case_one_transform(t):
                return make_tensor(t.shape, dtype=t.dtype, device=t.device, noncontiguous=True)
            _compare_out(_case_one_transform)

            def _case_two_transform(t):
                return make_tensor((0,), dtype=t.dtype, device=t.device)
            _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)
            out = _apply_out_transform(_case_two_transform, expected)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                op_out(out=out)
            for w in caught:
                if 'An output with one or more elements' in str(w.message):
                    self.fail('Resizing an out= argument with no elements threw a resize warning!')
            wrong_device = None
            if torch.device(device).type != 'cpu':
                wrong_device = 'cpu'
            elif torch.xpu.is_available():
                wrong_device = 'xpu'
            factory_fn_msg = '\n\nNOTE: If your op is a factory function (i.e., it accepts TensorOptions) you should mark its OpInfo with `is_factory_function=True`.'
            if wrong_device is not None:

                def _case_three_transform(t):
                    return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)
                out = _apply_out_transform(_case_three_transform, expected)
                if op.is_factory_function and sample.kwargs.get('device', None) is None:
                    op_out(out=out)
                else:
                    msg_fail = f'Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}.' + factory_fn_msg
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)
            _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
            if isinstance(expected, torch.Tensor) and expected.dtype in _dtypes or (not isinstance(expected, torch.Tensor) and any((t.dtype in _dtypes for t in expected))):

                def _case_four_transform(t):
                    return make_tensor(t.shape, dtype=torch.long, device=t.device)
                out = _apply_out_transform(_case_four_transform, expected)
                msg_fail = 'Expected RuntimeError when doing an unsafe cast!'
                msg_fail = (msg_fail if not isinstance(expected, torch.Tensor) else f'Expected RuntimeError when doing an unsafe cast from a result of dtype {expected.dtype} into an out= with dtype torch.long') + factory_fn_msg
                if op.is_factory_function and sample.kwargs.get('dtype', None) is None:
                    op_out(out=out)
                else:
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)

    @_variant_ops(op_db)
    def test_variant_consistency_eager(self, device, dtype, op):
        method = op.method_variant
        inplace = op.inplace_variant
        operator = op.operator_variant
        inplace_operator = op.inplace_operator_variant
        inplace_ops = [inplace, inplace_operator]
        variants = [method, inplace, operator, inplace_operator]
        operators = [operator, inplace_operator]
        for a_op in op.aliases:
            variants.append(a_op.op)
            variants.append(a_op.method_variant)
            variants.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)
        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants))
        operators = tuple(filter(None, operators))
        _requires_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad, include_conjugated_inputs=include_conjugated_inputs)
        samples = list(samples)

        def _test_consistency_helper(samples, variants):
            for sample in samples:
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None
                output_process_fn_grad = sample.output_process_fn_grad if sample.output_process_fn_grad else lambda x: x
                skip_inplace = False
                if isinstance(expected_forward, torch.Tensor) and expected_forward.dtype is not tensor.dtype:
                    skip_inplace = True
                if isinstance(expected_forward, torch.Tensor) and dtype in op.supported_backward_dtypes(torch.device(device).type):
                    output_process_fn_grad(expected_forward).sum().backward()
                    expected_grad = tensor.grad
                for variant in variants:
                    if variant in inplace_ops and skip_inplace:
                        continue
                    tensor.grad = None
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input
                    if variant in inplace_ops and sample.broadcasts_input:
                        with self.assertRaises(RuntimeError, msg='inplace variant either incorrectly allowed resizing or you have marked the sample {} incorrectly with `broadcasts_self=True'.format(sample.summary())):
                            variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                        continue
                    if variant in operators and sample.kwargs:
                        continue
                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    self.assertEqual(expected_forward, variant_forward)
                    if expected_grad is not None and (variant not in inplace_ops or op.supports_inplace_autograd):
                        output_process_fn_grad(variant_forward).sum().backward()
                        self.assertEqual(expected_grad, tensor.grad)
        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                skip_inplace = False
                if isinstance(expected_forward, torch.Tensor) and expected_forward.dtype is not tensor.dtype:
                    skip_inplace = True
                if skip_inplace:
                    return
                for variant in variants:
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input
                    inp_tensor = cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    data_ptr = inp_tensor.data_ptr()
                    if variant in operators and sample.kwargs:
                        continue
                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    if isinstance(variant_forward, torch.Tensor):
                        self.assertEqual(data_ptr, variant_forward.data_ptr(), atol=0, rtol=0)
                    else:
                        self.assertTrue(False, 'Non-tensor outputs for inplace ops are not supported')
        if len(inplace_ops) > 0:
            inplace_samples = list(filter(lambda sample: not sample.broadcasts_input, samples))
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)

    @ops(op_db, allowed_dtypes=(torch.complex32,))
    def test_complex_half_reference_testing(self, device, dtype, op):
        if not op.supports_dtype(torch.complex32, device):
            unittest.skip('Does not support complex32')
        for sample in op.sample_inputs(device, dtype):
            actual = op(sample.input, *sample.args, **sample.kwargs)
            transformed_sample = sample.transform(lambda x: x.to(torch.complex64) if isinstance(x, torch.Tensor) and x.dtype is torch.complex32 else x)
            expected = op(transformed_sample.input, *transformed_sample.args, **transformed_sample.kwargs)
            expected = tree_map(lambda x: x.to(torch.complex32) if isinstance(x, torch.Tensor) and x.dtype is torch.complex64 else x, expected)
            self.assertEqual(actual, expected, exact_dtype=False)

    @ops(op_db, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, 'Test uses undefined behavior')
    def test_non_standard_bool_values(self, device, dtype, op):

        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x
            true_vals = torch.randint(2, 255, x.shape, dtype=torch.uint8, device=x.device)
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)
            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret
        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)
            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)
            self.assertEqual(expect, actual)

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @skipMeta
    @onlyNativeDeviceTypes
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        device_type = torch.device(device).type
        include_complex32 = (torch.complex32,) if op.supports_dtype(torch.complex32, device_type) else ()
        allowed_backward_dtypes = floating_and_complex_types_and(*(torch.half, torch.bfloat16) + include_complex32)
        supported_dtypes = set()
        unsupported_dtypes = set()
        supported_backward_dtypes = set()
        unsupported_backward_dtypes = set()

        def unsupported(dtype):
            unsupported_dtypes.add(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.add(dtype)
        for dtype in all_types_and_complex_and(*(torch.half, torch.bfloat16, torch.bool) + include_complex32):
            requires_grad = dtype in allowed_backward_dtypes
            try:
                samples = tuple(op.sample_inputs(device, dtype, requires_grad=requires_grad))
            except Exception as e:
                unsupported(dtype)
                continue
            for sample in samples:
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                    supported_dtypes.add(dtype)
                except Exception as e:
                    unsupported(dtype)
                    continue

                def _tensor_requires_grad(x):
                    if isinstance(x, dict):
                        for (k, v) in x.items():
                            if _tensor_requires_grad(v):
                                return True
                    if isinstance(x, (list, tuple)):
                        for a in x:
                            if _tensor_requires_grad(a):
                                return True
                    if isinstance(x, torch.Tensor) and x.requires_grad:
                        return True
                    return False
                requires_grad = _tensor_requires_grad(sample.input) or _tensor_requires_grad(sample.args) or _tensor_requires_grad(sample.kwargs)
                if not requires_grad:
                    continue
                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(result[0], torch.Tensor):
                        backward_tensor = result[0]
                    else:
                        continue
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    supported_backward_dtypes.add(dtype)
                except Exception as e:
                    unsupported_backward_dtypes.add(dtype)
        supported_forward = supported_dtypes - unsupported_dtypes
        partially_supported_forward = supported_dtypes & unsupported_dtypes
        unsupported_forward = unsupported_dtypes - supported_dtypes
        supported_backward = supported_backward_dtypes - unsupported_backward_dtypes
        partially_supported_backward = supported_backward_dtypes & unsupported_backward_dtypes
        unsupported_backward = unsupported_backward_dtypes - supported_backward_dtypes
        device_type = torch.device(device).type
        claimed_forward = set(op.supported_dtypes(device_type))
        supported_but_unclaimed_forward = supported_forward - claimed_forward
        claimed_but_unsupported_forward = claimed_forward & unsupported_forward
        claimed_backward = set(op.supported_backward_dtypes(device_type))
        supported_but_unclaimed_backward = supported_backward - claimed_backward
        claimed_but_unsupported_backward = claimed_backward & unsupported_backward
        if len(partially_supported_forward) + len(partially_supported_backward) > 0:
            msg = 'Some dtypes for {0} on device type {1} are only partially supported!\n'.format(op.name, device_type)
            if len(partially_supported_forward) > 0:
                msg = msg + 'The following dtypes only worked on some samples during forward: {0}.\n'.format(partially_supported_forward)
            if len(partially_supported_backward) > 0:
                msg = msg + 'The following dtypes only worked on some samples during backward: {0}.\n'.format(partially_supported_backward)
            print(msg)
        if len(supported_but_unclaimed_forward) + len(claimed_but_unsupported_forward) + len(supported_but_unclaimed_backward) + len(claimed_but_unsupported_backward) == 0:
            return
        if op in python_ref_db:
            if len(claimed_but_unsupported_forward) + len(claimed_but_unsupported_backward) == 0:
                return
        msg = 'The supported dtypes for {0} on device type {1} are incorrect!\n'.format(op.name, device_type)
        if len(supported_but_unclaimed_forward) > 0:
            msg = msg + 'The following dtypes worked in forward but are not listed by the OpInfo: {0}.\n'.format(supported_but_unclaimed_forward)
        if len(supported_but_unclaimed_backward) > 0:
            msg = msg + 'The following dtypes worked in backward but are not listed by the OpInfo: {0}.\n'.format(supported_but_unclaimed_backward)
        if len(claimed_but_unsupported_forward) > 0:
            msg = msg + 'The following dtypes did not work in forward but are listed by the OpInfo: {0}.\n'.format(claimed_but_unsupported_forward)
        if len(claimed_but_unsupported_backward) > 0:
            msg = msg + 'The following dtypes did not work in backward but are listed by the OpInfo: {0}.\n'.format(claimed_but_unsupported_backward)
        self.fail(msg)

class TestCompositeCompliance(TestCase):

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_with_mode(op, args, kwargs, self.assertEqual)
            composite_compliance.check_all_permutations(op, args, kwargs, self.assertEqual)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_backward_formula(op.get_op(), args, kwargs, sample.output_process_fn_grad, op.gradcheck_wrapper, self.assertEqual)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_forward_ad(self, device, dtype, op):
        if torch.float not in op.supported_backward_dtypes(device):
            raise unittest.SkipTest('Does not support autograd')
        if not op.supports_forward_ad:
            raise unittest.SkipTest('Does not support forward_ad')
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_forward_ad_formula(op.get_op(), args, kwargs, op.gradcheck_wrapper, self.assertEqual)

@skipIfSlowGradcheckEnv
class TestMathBits(TestCase):

    def _test_math_view(self, device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, out_type):
        inplace_variant = op.inplace_variant

        def clone_and_perform_view(input, **kwargs):
            if isinstance(input, torch.Tensor):
                requires_grad = kwargs.get('requires_grad', input.requires_grad)
                with torch.no_grad():
                    input = math_op_physical(input)
                input = math_op_view(input)
                assert input.is_leaf
                return input.requires_grad_(requires_grad)
            if isinstance(input, Sequence):
                out = list(map(clone_input_helper, input))
                out[0] = clone_and_perform_view(out[0])
                return tuple(out)
        for sample in samples:
            tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
            cloned1 = clone_and_perform_view(sample.input)
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)
            forward_with_mathview = op(cloned1, *sample.args, **sample.kwargs)
            self.assertEqual(expected_forward, forward_with_mathview)
            if inplace_variant is not None and (not sample.broadcasts_input):
                cloned2 = clone_and_perform_view(tensor, requires_grad=False)
                if isinstance(expected_forward, torch.Tensor) and expected_forward.dtype is tensor.dtype:
                    inplace_forward = inplace_variant(cloned2, *sample.args, **sample.kwargs)
                    self.assertTrue(is_bit_set(inplace_forward))
                    self.assertEqual(inplace_forward, expected_forward)
            if isinstance(expected_forward, torch.Tensor) and expected_forward.requires_grad:
                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)
                expected_forward = output_process_fn_grad(expected_forward)
                forward_with_mathview = output_process_fn_grad(forward_with_mathview)
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                expected_forward.sum().backward(retain_graph=True)
                forward_with_mathview.sum().backward(retain_graph=True)
                if tensor.grad is not None:
                    cloned1_tensor = cloned1 if isinstance(cloned1, torch.Tensor) else cloned1[0]
                    self.assertEqual(tensor.grad, cloned1_tensor.grad)
                    (tensor.grad, cloned1_tensor.grad) = (None, None)
                    if out_type(expected_forward):
                        grad = torch.randn_like(expected_forward)
                        expected_forward.backward(grad)
                        forward_with_mathview.backward(math_op_view(math_op_physical(grad)))
                        self.assertEqual(tensor.grad, cloned1_tensor.grad)

    @ops(ops_and_refs, allowed_dtypes=(torch.cfloat,))
    def test_conj_view(self, device, dtype, op):
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")
        math_op_physical = torch.conj_physical
        math_op_view = torch.conj
        _requires_grad = torch.cfloat in op.supported_backward_dtypes(torch.device(device).type)
        is_bit_set = torch.is_conj
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, torch.is_complex)

    @ops(ops_and_refs, allowed_dtypes=(torch.double,))
    def test_neg_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest('Operation not tested with tensors with negative bit.')
        math_op_physical = torch.neg
        math_op_view = torch._neg_view
        is_bit_set = torch.is_neg
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, lambda x: True)

    @ops(ops_and_refs, allowed_dtypes=(torch.cdouble,))
    def test_neg_conj_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest('Operation not tested with tensors with negative bit.')
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        def math_op_physical(x):
            return -x.conj_physical()

        def math_op_view(x):
            return torch._neg_view(x).conj()

        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)
        _requires_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        samples = itertools.islice(samples, 1)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, torch.is_complex)

def check_inplace_view(func, input, rs, input_size, input_strides):
    if func is None:
        return
    if isinstance(rs, torch.Tensor) and rs is input:
        unequal_size = rs.size() != input_size
        unequal_strides = rs.stride() != input_strides
        if unequal_size or unequal_strides:
            if isinstance(func, torch._ops.OpOverloadPacket):
                func = func.default
            if func is not torch.ops.aten.resize_.default:
                assert torch.Tag.inplace_view in func.tags

@skipIfSlowGradcheckEnv
class TestTagsMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(args[0], torch.Tensor):
            old_size = args[0].size()
            old_stride = args[0].stride()
            rs = func(*args, **kwargs)
            check_inplace_view(func, args[0], rs, old_size, old_stride)
        else:
            rs = func(*args, **kwargs)
        return rs

@skipIfSlowGradcheckEnv
class TestTags(TestCase):

    @onlyCPU
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    def test_tags(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            input = sample.input
            if isinstance(input, torch.Tensor):
                old_size = input.size()
                old_stride = input.stride()
                with TestTagsMode():
                    rs = op(input, *sample.args, **sample.kwargs)
                aten_name = op.aten_name if op.aten_name is not None else op.name
                opoverloadpacket = getattr(torch.ops.aten, aten_name, None)
                check_inplace_view(opoverloadpacket, input, rs, old_size, old_stride)

@skipIfSlowGradcheckEnv
class TestRefsOpsInfo(TestCase):
    import_paths = ['_refs', '_refs.special', '_refs.nn.functional', '_refs.fft']
    module_alls = [(path, import_module(f'torch.{path}').__all__) for path in import_paths]
    ref_ops_names = tuple(itertools.chain.from_iterable(([f'{path}.{op}' for op in module_all] for (path, module_all) in module_alls)))
    ref_db_names = set((ref_op.name for ref_op in python_ref_db))
    skip_ref_ops = {'_refs.bitwise_right_shift', '_refs.copy_to', '_refs.empty_strided', '_refs.equal', '_refs.full', '_refs.full_like', '_refs.item', '_refs.to', '_refs.ones', '_refs.ones_like', '_refs.std_var', '_refs.swap_axes', '_refs.uniform', '_refs.scalar_tensor', '_refs.trunc_divide', '_refs.zeros', '_refs.zeros_like', '_refs.rfloordiv', '_refs.rtruediv', '_refs.rpow', '_refs.index_add_', '_refs.index_copy_', '_refs.index_fill_'}
    not_in_decomp_table = {'_refs.nn.functional.elu', '_refs.nn.functional.mse_loss', '_refs.var', '_refs.rsub', '_refs.index_add_', '_refs.broadcast_shapes', '_refs.broadcast_tensors', '_refs.nn.functional.tanhshrink', '_refs.rfloordiv', '_refs.rtruediv', '_refs.rpow', '_refs.allclose', '_refs.atleast_1d', '_refs.atleast_2d', '_refs.atleast_3d', '_refs.broadcast_to', '_refs.chunk', '_refs.column_stack', '_refs.contiguous', '_refs.dsplit', '_refs.dstack', '_refs.fill', '_refs.flatten', '_refs.fliplr', '_refs.flipud', '_refs.float_power', '_refs.hsplit', '_refs.hstack', '_refs.isclose', '_refs.isfinite', '_refs.isreal', '_refs.movedim', '_refs.narrow', '_refs.nn.functional.l1_loss', '_refs.nn.functional.poisson_nll_loss', '_refs.positive', '_refs.ravel', '_refs.reshape', '_refs.square', '_refs.tensor_split', '_refs.to', '_refs.true_divide', '_refs.trunc_divide', '_refs.vsplit', '_refs.vstack', '_refs.linalg.matrix_norm', '_refs.linalg.norm', '_refs.linalg.svd', '_refs.linalg.svdvals', '_refs.unflatten', '_refs.sum_to_size', '_refs.full', '_refs.full_like', '_refs.ones_like', '_refs.round', '_refs.scalar_tensor', '_refs.zeros_like', '_refs.expand_as', '_refs.as_strided', '_refs.copy_to', '_refs.equal', '_refs.conj', '_refs.real', '_refs.imag'}

    @parametrize('op', ref_ops_names)
    def test_refs_are_in_python_ref_db(self, op):
        if op in self.skip_ref_ops:
            raise unittest.SkipTest(f'{op} does not have an entry in python_ref_db')
        self.assertIn(op, self.ref_db_names)

    @parametrize('op', ref_ops_names)
    def test_refs_are_in_decomp_table(self, op):
        path = op.split('.')
        module_path = '.'.join(path[:-1])
        op_name = path[-1]
        op_impl = getattr(import_module(f'torch.{module_path}'), op_name)
        if op in self.not_in_decomp_table:
            self.assertNotIn(op_impl, torch._decomp.decomposition_table.values(), f'Unexpectedly found {op} in torch._decomp.decomposition_table.values()')
        else:
            self.assertIn(op_impl, torch._decomp.decomposition_table.values(), f'Did not find {op} in torch._decomp.decomposition_table.values()')
fake_skips = ('aminmax', 'cholesky', 'cholesky_inverse', 'cov', 'istft', 'linalg.eigvals', 'linalg.eigvalsh', 'linalg.matrix_power', 'linalg.matrix_rank.hermitian', 'linalg.pinv.hermitian', 'linalg.solve', 'linalg.tensorsolve', 'lu_solve', 'multinomial', 'mvlgamma.mvlgamma_p_1', 'mvlgamma.mvlgamma_p_3', 'mvlgamma.mvlgamma_p_5', 'nanmean', 'quantile', 'nanquantile', 'nn.functional.ctc_loss', 'nn.functional.embedding_bag', 'nn.functional.nll_loss', 'nn.functional.max_pool1d', 'to_sparse', 'tensor_split', 'repeat_interleave', 'segment_reduce.lengths', 'sparse.sampled.addmm', 'nn.functional.one_hot', 'narrow')
fake_autocast_device_skips = defaultdict(dict)
fake_autocast_device_skips['cpu'] = set(('linalg.pinv',))
dynamic_output_op_tests = ('argwhere', 'bincount', 'combinations', 'linalg.lstsq', 'masked_select', 'nonzero', 'unique_consecutive', 'unique', 'linalg.lstsq.grad_oriented')
sometimes_dynamic_output_op_test = ('__getitem__', 'index_select')
data_dependent_op_tests = ('equal', 'corrcoef', 'nn.functional.gaussian_nll_loss', 'allclose')
aliasing_failures = ('histogramdd', 'nn.functional.pixel_shuffle', 'nn.functional.pixel_unshuffle')
fake_tensor_stride_failing_ops = {'fft.fft2', 'fft.fft', 'fft.fftn', 'fft.hfft2', 'fft.hfft', 'fft.hfftn', 'fft.ifft2', 'fft.ifft', 'fft.ifftn', 'fft.ihfft2', 'fft.ihfft', 'fft.ihfftn', 'fft.irfft2', 'fft.irfft', 'fft.irfftn', 'fft.rfft2', 'fft.rfft', 'fft.rfftn', 'svd', 'linalg.svd'}
fake_backward_xfails = fake_tensor_stride_failing_ops | {'linalg.cond', 'linalg.matrix_norm', 'linalg.norm', 'linalg.svd', 'linalg.svdvals', 'nn.functional.binary_cross_entropy_with_logits', 'nn.functional.huber_loss', 'nn.functional.logsigmoid', 'nn.functional.multilabel_soft_margin_loss', 'pca_lowrank', 'roll', 'svd_lowrank', 'sgn', 'cholesky', 'linalg.eigh', 'symeig'}
fake_backward_xfails = {xfail(stride_skip) for stride_skip in fake_backward_xfails} | {xfail('segment_reduce', 'lengths'), xfail('norm', 'nuc'), xfail('linalg.norm', 'subgradients_at_zero'), skip('nn.functional.ctc_loss')}
fake_autocast_backward_xfails = {skip('nn.functional.binary_cross_entropy'), skip('sparse.sampled_addmm'), skip('linalg.pinv'), skip('linalg.pinv', 'hermitian'), skip('linalg.pinv', 'singular'), skip('pinverse')}

@skipIfSlowGradcheckEnv
class TestFakeTensor(TestCase):

    def _test_fake_helper(self, device, dtype, op, context):
        name = op.name
        if op.variant_test_name:
            name += '.' + op.variant_test_name
        if name in fake_skips or 'sparse' in name or 'jiterator' in name:
            self.skipTest('Skip failing test')
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            try:
                mode = FakeTensorMode(throw_on_data_dependent_ops=True)

                def map_to_fake(e):
                    if isinstance(e, torch.Tensor):
                        return mode.from_tensor(e)
                    else:
                        return e
                input = tree_map(map_to_fake, sample.input)
                args = tree_map(map_to_fake, sample.args)
                kwargs = tree_map(map_to_fake, sample.kwargs)
                try:
                    with context():
                        res = op(sample.input, *sample.args, **sample.kwargs)
                except Exception as e:
                    continue
                with context():
                    with mode:
                        res_fake = op(input, *args, **kwargs)
                for (fake_out, real_out) in zip(tree_flatten(res_fake)[0], tree_flatten(res)[0]):
                    if not isinstance(fake_out, torch.Tensor):
                        self.assertTrue(not isinstance(real_out, torch.Tensor))
                        continue
                    self.assertTrue(isinstance(fake_out, FakeTensor))
                    check_strides = name not in fake_tensor_stride_failing_ops
                    prims.utils.compare_tensor_meta(fake_out, real_out, check_strides)
                    if name not in aliasing_failures:
                        fake_aliasing = outputs_alias_inputs((input, args, kwargs), res_fake)
                        real_aliasing = outputs_alias_inputs((sample.input, sample, args, sample.kwargs), res)
                        self.assertEqual(fake_aliasing, real_aliasing)
                self.assertTrue(name not in dynamic_output_op_tests and name not in data_dependent_op_tests)
            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                pass
            except torch._subclasses.fake_tensor.DynamicOutputShapeException:
                self.assertTrue(name in dynamic_output_op_tests or name in sometimes_dynamic_output_op_test)
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                self.assertTrue(name in data_dependent_op_tests)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        self._test_fake_helper(device, dtype, op, contextlib.nullcontext)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake_autocast(self, device, dtype, op):
        if op.name in fake_autocast_device_skips[device]:
            self.skipTest('Skip failing test')
        context = torch.xpu.amp.autocast if device == 'xpu' else torch.cpu.amp.autocast
        self._test_fake_helper(device, dtype, op, context)

    def _test_fake_crossref_helper(self, device, dtype, op, context):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for (iter, sample) in enumerate(samples):
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            common_skip_ops = (aten.detach.default, aten.empty_strided.default, aten.copy_.default, aten.is_same_size.default)
            with torch._subclasses.CrossRefFakeMode(ignore_op_fn=lambda fn: fn in common_skip_ops, check_aliasing=True):
                with warnings.catch_warnings(), context():
                    composite_compliance.compute_expected_grads(op.get_op(), args, kwargs, sample.output_process_fn_grad, op.gradcheck_wrapper)

    @skipIfRocm
    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps('TestFakeTensor', 'test_fake_crossref_backward_no_amp', fake_backward_xfails)
    def test_fake_crossref_backward_no_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, contextlib.nullcontext)

    @skipIfRocm
    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps('TestFakeTensor', 'test_fake_crossref_backward_amp', fake_backward_xfails | fake_autocast_backward_xfails)
    def test_fake_crossref_backward_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, torch.xpu.amp.autocast)
instantiate_device_type_tests(TestCommon, globals())
instantiate_device_type_tests(TestCompositeCompliance, globals())
instantiate_device_type_tests(TestMathBits, globals())
instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for='cpu')
instantiate_device_type_tests(TestFakeTensor, globals())
instantiate_device_type_tests(TestTags, globals())
if __name__ == '__main__':
    run_tests()