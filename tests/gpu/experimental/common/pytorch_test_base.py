import collections
import re
import gc
import contextlib
import os
import inspect
import functools

import torch
import intel_extension_for_pytorch
import logging
from functools import wraps
from torch.testing._internal.common_device_type import DeviceTypeTestBase
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch.testing._internal.common_utils import IS_WINDOWS
import unittest
from typing import cast, List, Optional, Tuple, Union

TEST_XPU = torch.xpu.is_available()
TEST_MULTIGPU = TEST_XPU and torch.xpu.device_count() >= 2

RUN_XPU = TEST_XPU
RUNXPU_MULTI_GPU = TEST_MULTIGPU
RUN_XPU_HALF = RUN_XPU

PYTORCH_XPU_MEMCHECK = os.getenv('PYTORCH_XPU_MEMCHECK', '0') == '1'
TEST_SKIP_XPU_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_SKIP_XPU_MEM_LEAK_CHECK', '0') == '1'

DEFAULT_FLOATING_PRECISION = 1e-3

log = logging.getLogger(__name__)
TORCH_TEST_PRECISIONS = {
    # test_name : floating_precision,
    'test_pow_xpu_float32': 0.0035,
    'test_pow_xpu_float64': 0.0045,
    'test_var_neg_dim_xpu_bfloat16': 0.01,
    'test_sum_xpu_bfloat16': 0.1,
}

DISABLED_TORCH_TESTS_ANY = {
    # empty
}

DISABLED_TORCH_TESTS_XPU_ONLY = {
    # to be added
    # "TestBinaryUfuncsXPU": {
    #     "test_cumulative_trapezoid",    # core dumped ... Floating point exception -> cumsum issue @Yu, Kevin
    #     "test_int_and_float_pow",   # core dumped ... free(): invalid size -> KernelPointWiseApply2
    # },
    # "TestForeachXPU": {
    #     "test_binary_op_scalar_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    #     "test_binary_op_scalarlist_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    #     "test_binary_op_tensorlists_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    #     "test_minmax_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    #     "test_pointwise_op_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    #     "test_unary_slowpath",  # core dumped ... munmap_chunk(): invalid pointer -> torch.profiler.ProfilerActivity.XPU
    # },
    # "TestAutogradDeviceTypeXPU": {
    #     "test_cdist_same_inputs",   # too slow    -> no slow in AOT
    #     "test_cdist",   # too slow   -> no slow in AOT
    #     "test_sparse_backward",     # core dumped ... Segmentation fault -> no sparse tensor math op.
    # },
    # "TestTorchDeviceTypeXPU": {
        # "test_cdist_large_batch",   # too slow  -> not slow with AOT. But `-fno-sycl-id-queries-fit-in-int' to disable range check. -30 (CL_INVALID_VALUE)
        # "test_cdist_large",     # too slow -> not slow with AOT. But `-fno-sycl-id-queries-fit-in-int' to disable range check. -30 (CL_INVALID_VALUE)
        # "test_conv_transposed_large",   # too slow  -> <deconvolution> start barrier(606us) submit(10387us) end barrier(122us) event wait(75568616us) event_duration_0(74985075us) total(75579744us)
        # "test_cov",     # core dumped ... free(): invalid size  -> accurate issue.
        # "test_dim_function_empty",  # core dumped ... Floating point exception -> cumsum issue in handling numel or dim = 0. Should be fixed by Kevin.
        # "test_index_select",    # core dumped ... munmap_chunk(): invalid pointer  -> RuntimeError: index_select(): Expected dtype int64 for index but got: Int
        # "test_large_cumprod",   # too slow  -> OOM on my gen9 host
        # "test_large_cumsum",   # too slow -> OOM on my gen9 host
    # },
    # "TestReductionsXPU": {
        # "test_mode",    # too slow -> no too slow in AOT
        # "test_nansum_out_dtype",    # too slow
        # "test_nansum",  # too slow
        # "test_noncontiguous_all",   # too slow
        # "test_noncontiguous_expanded",  # too slow
        # "test_noncontiguous_innermost",     # too slow
        # "test_noncontiguous_outermost",     # too slow
        # "test_noncontiguous_transposed",    # too slow
        # "test_numpy_named_args",    # too slow  -> passed
        # "test_prod_bool",   # too slow
        # "test_prod_gpu",    # too slow  -> passed
        # "test_quantile_backward",   # too slow
        # "test_quantile_error",  # too slow  -> passed
        # "test_quantile",    # too slow
        # "test_reduction_empty_any_all",     # too slow
        # "test_reduction_split",     # too slow
        # "test_reduction_vectorize_along_input_corner",  # too slow
        # "test_reduction_vectorize_along_output",  # too slow -> passed
        # "test_std_correction_vs_numpy",     # too slow
        # "test_std_mean_all_dims",   # too slow -> passed
        # "test_std_mean_correction",     # too slow
        # "test_std_mean_some_dims",  # too slow -> passed
        # "test_std_mean",    # too slow
        # "test_std_vs_numpy",    # too slow
        # "test_sum_dim_reduction_uint8_overflow",    # too slow
        # "test_sum_noncontig",   # too slow
        # "test_sum_vs_numpy",    # too slow
        # "test_tensor_reduce_ops_empty",     # too slow
        # "test_var_correction_vs_numpy",     # too slow
        # "test_var_large_input",     # too slow
        # "test_var_mean_all_dims",   # too slow
        # "test_var_mean_correction",     # too slow
        # "test_var_mean_some_dims",  # too slow
        # "test_var_mean",    # too slow
        # "test_var_stability2",  # too slow
        # "test_var_stability",   # too slow
        # "test_var_unbiased",    # too slow
        # "test_var_vs_numpy",    # too slow
    #     "test_var",     # too slow
    # },
    # "TestUnaryUfuncsXPU": {
    #     "test_batch_vs_slicing",   # too slow
    #     "test_frexp_out",    # core dumped ... free(): invalid size  -> OOM ../neo/opencl/source/os_interface/linux/drm_command_stream.inl
    #     "test_out_arg_all_dtypes",   # core dumped ... Segmentation fault
    # },
    # "TestCommonXPU": {
        # "test_dtypes",  # core dumped ... munmap_chunk(): invalid pointer
        # "test_out",     # core dumped ... free(): invalid size  -> oneDNN assert
        # "test_variant_consistency_eager",   # core dumped ... munmap_chunk(): invalid pointer
        # "test_multiple_devices",    # multi device not ready   -> issue in test scripts
    # },
    # "TestGradientsXPU": {
        # "test_fn_grad", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
        # "test_fn_gradgrad", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
        # "test_forward_mode_AD", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
        # "test_inplace_forward_mode_AD", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
        # "test_inplace_grad", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
        # "test_inplace_gradgrad", # core dumped ... munmap_chunk(): invalid pointer  -> OOM on my gen9 host
    # },
    # "TestJitXPU": {
    #     "test_variant_consistency_jit", # core dumped ... munmap_chunk(): invalid pointer
    # },
    # "TestMathBitsXPU": {
        # "test_conj_view",   # core dumped ... munmap_chunk(): invalid pointer
        # "test_neg_view",   # core dumped ... munmap_chunk(): invalid pointer
    # },
    # "TestSparseCSRXPU": {
        # "test_add",     # core dumped ... Floating point exception
        # "test_coo_to_csr_convert",     # core dumped ... Segmentation fault
        # "test_csr_matvec",     # core dumped ... Floating point exception
        # "test_matmul_device_mismatch",  # core dumped ... Segmentation fault
        # "test_mm",      # core dumped ... Floating point exception
        # "test_sparse_addmm",    # core dumped ... Floating point exception
        # "test_sparse_mm",   # core dumped ... Floating point exception
    # },
    # "TestNNDeviceTypeXPU": {
        # "test_conv_ndhwc",  # core dumped ... Segmentation fault
        # "test_conv_nhwc",   # core dumped ... Segmentation fault
        # "test_conv_transposed_large",   # core dumped ... Floating point exception
        # "test_grid_sample_large_index_2d",  # too slow
        # "test_grid_sample_large_index_3d",  # too slow
        # "test_softmax_64bit_indexing",  # too slow
        # "test_softmax_results",     # core dumped ... Floating point exception
    # },
    "TestFFTXPU": {
        # 'test_batch_istft', # failures
        'test_fft2_fftn_equivalence',   # hang
        'test_fft2_numpy',  # hang
        'test_fft_input_modification',  # hang
        # 'test_fft_ifft_rfft_irfft',     # failures
        # 'test_fft_round_trip',  # failures
        # 'test_fft_type_promotion',  # pass
        'test_fftn_round_trip',     # hang
        # 'test_istft_of_sine',   # failures
        # 'test_istft_round_trip_simple_cases',   # failures
        # 'test_istft_round_trip_various_params',     # failures
        # 'test_istft_round_trip_with_padding',   # failures
        # 'test_istft_throws',    # pass
        # 'test_reference_1d',    # failures
        'test_reference_nd',    # hang
    },
}


def tf32_is_not_fp32():
    if not torch.xpu.is_available():
        return False
    return True

@contextlib.contextmanager
def tf32_off():
    # FixMe: we can't support tf32 right now
    yield

@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-4):
    # FixMe: we can't support tf32 right now
    old_allow_tf32_matmul = False
    old_precision = self.precision

    self.precision = tf32_precision
    try:
        yield
    finally:
        self.precision = old_precision

def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            cond = tf32_is_not_fp32()
            if 'device' in kwargs:
                cond = cond and (torch.device(kwargs['device']).type == 'xpu')
            if 'dtype' in kwargs:
                cond = cond and (kwargs['dtype'] in {torch.float32, torch.complex64})
            if cond:
                with_tf32_disabled(kwargs['self'], lambda: f(**kwargs))
                with_tf32_enabled(kwargs['self'], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped
    return wrapper

def with_tf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with ft32_off():
            return f(*args, **kwargs)

    return wrapped



__xpu_ctx_rng_initialized = False

def initialize_xpu_context_rng():
    global __xpu_ctx_rng_initialized
    assert TEST_XPU, 'XPU must be available when calling initialized_xpu_context_rng'
    if not __xpu_ctx_rng_initialized:
        for i in range(torch.xpu.device_count()):
            torch.randn(1, device="xpu:{}".format(i))
        __xpu_ctx_rng_initialized = True

class XPUMemoryLeakCheck():
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase
        initialize_xpu_context_rng()

    @staticmethod
    def get_xpu_memory_usage():
        num_devices = torch.xpu.device_count()
        gc.collect()
        return tuple(torch.xpu.memory_allocated(i) for i in range(num_devices))

    def __enter__(self):
        self.befores = self.get_xpu_memory_usage()

    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            return

        afters = self.get_xpu_memory_usage()

        for i, (before, after) in enumerate(zip(self.befores, afters)):
            self.testcase.assertEqual(
                before, after, msg='{} leaked {} bytes XPU memory on device {}'.format(
                    self.name, after - before, i))

class XPUNonDefaultStream():
    def __enter__(self):
        beforeDevice = torch.xpu.current_device()
        self.beforeStreams = []
        for d in range(torch.xpu.device_count()):
            self.beforeStreams.append(torch.xpu.current_stream(d))
            deviceStream = torch.xpu.Stream(device=d)
            intel_extension_for_pytorch._C._setCurrentStream(deviceStream._cdata)
        torch.xpu.set_device(beforeDevice)

    def __exit__(self, exec_type, exec_value, traceback):
        beforeDevice = torch.xpu.current_device()
        for d in range(torch.xpu.device_count()):
            intel_extension_for_pytorch._C._setCurrentStream(self.beforeStreams[d]._cdata)
        torch.xpu.set_device(beforeDevice)

def skipXPUMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_xpu_memory_leak_check', True):
            fn._do_xpu_memory_leak_check = not condition
        return fn
    return dec

def skipXPUNonDefaultStreamIf(condition):
    def dec(fn):
        if getattr(fn, '_do_xpu_non_default_stream', True):
            fn._do_xpu_non_default_stream = not condition
        return fn
    return dec

class TestCase(TorchTestCase):

    def _should_stop_test_suite(self):
        return False

        # FixMe: torch.xpu._is_initialized() not implemented
        if torch.xpu._is_initialized():
            try:
                torch.xpu.synchronize()
            except RuntimeError as rte:
                return True
            return False
        else:
            return False

    _do_xpu_memory_leak_check = False
    _do_xpu_non_default_stream = False

    _ignore_not_implemented_error = True

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)

        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do XPU memory check.
            if not TEST_SKIP_XPU_MEM_LEAK_CHECK:
                self._do_xpu_memory_leak_check &= getattr(test_method, '_do_xpu_memory_leak_check', True)
                if self._do_xpu_memory_leak_check and not IS_WINDOWS:
                    self.wrap_with_xpu_policy(method_name, self.assertLeaksNoXPUTensors)

            # Wraps the tested method if we should enforce non default XPU stream.
            self._do_xpu_non_default_stream &= getattr(test_method, '_do_xpu_non_default_stream', True)
            if self._do_xpu_non_default_stream and not IS_WINDOWS:
                self.wrap_with_xpu_policy(method_name, self.enforceNonDefaultStream)

    def assertLeaksNoXPUTensors(self, name=None):
        name = self.id() if name is None else name
        return XPUMemoryLeakCheck(self, name)

    def enforceNonDefaultStream(self):
        return XPUNonDefaultStream()

    def wrap_with_xpu_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        fullname = self.id().lower()
        if TEST_XPU and ('gpu' in fullname or 'xpu' in fullname):
            setattr(self, method_name, super().wrap_method_with_policy(test_method, policy))

    def run(self, result=None):
        super().run(result=result)
        if self._should_stop_test_suite():
            result.stop()

class MatchSet(object):

    def __init__(self):
        self.exact = set()
        self.regex = set()


def prepare_match_set(s):
    ps = dict()
    for k, v in s.items():
        mset = MatchSet()
        for m in v:
            if re.match(r'\w+$', m):
                mset.exact.add(m)
            else:
                mset.regex.add(m)
        ps[k] = mset
    return ps


def match_name(name, mset):
    if not mset:
        return False
    if name in mset.exact:
        return True
    for m in mset.regex:
        if re.match(m, name):
            return True
    return False


def union_of_enabled_tests(sets):
    union = collections.defaultdict(set)
    for s in sets:
        for k, v in s.items():
            union[k] = union[k] | v
    return union

DISABLED_TORCH_TESTS_XPU = union_of_enabled_tests(
    [DISABLED_TORCH_TESTS_ANY, DISABLED_TORCH_TESTS_XPU_ONLY])

DISABLED_TORCH_TESTS = {
    'CPU': prepare_match_set(DISABLED_TORCH_TESTS_ANY),
    'XPU': prepare_match_set(DISABLED_TORCH_TESTS_XPU),
}

class DPCPPTestBase(DeviceTypeTestBase):
    device_type = 'xpu'
    unsupported_dtypes = {
        torch.complex,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.cdouble,
        torch.cfloat,
    }
    primary_device = ''
    precision = DEFAULT_FLOATING_PRECISION

    @staticmethod
    def _alt_lookup(d, keys, defval):
        for k in keys:
            value = d.get(k, None)
            if value is not None:
                return value
            return defval

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.xpu.device_count()

        prim_device = cls.get_primary_device()
        xpu_str = 'xpu:{0}'
        non_primary_devices = [xpu_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # Acquires the current device as the primary (test) device
        cls.primary_device = 'xpu:{0}'.format(torch.xpu.current_device())

    # Overrides to instantiate tests that are known to run quickly
    # and correctly on XPU.

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        test_name = name + '_' + cls.device_type
        reason: str = "not ready on XPU"
        class_name = cls.__name__
        real_device_type = cls.device_type.upper()
        assert real_device_type in DISABLED_TORCH_TESTS, 'Unsupported device type:' + real_device_type
        disabled_torch_tests = DISABLED_TORCH_TESTS[real_device_type]

        @wraps(test)
        def disallowed_test(self, test=test, reason=reason):
            raise unittest.SkipTest(reason)
            return test(self, cls.device_type)

        if (match_name(test_name, disabled_torch_tests.get(class_name)) or
                match_name(name, disabled_torch_tests.get(class_name))):
            assert not hasattr(
                cls, test_name), 'Redefinition of test {0}'.format(test_name)
            setattr(cls, reason, "test")
            setattr(cls, test_name, disallowed_test)
        else:  # Test is allowed
            dtype_combinations = cls._get_dtypes(test)
            if dtype_combinations is None:  # Tests without dtype variants are instantiated as usual
                super().instantiate_test(name, copy.deepcopy(test), generic_cls=generic_cls)
            else:  # Tests with dtype variants have unsupported dtypes skipped
                # Sets default precision for floating types to bfloat16 precision
                if not hasattr(test, 'precision_overrides'):
                    test.precision_overrides = {}
                xpu_dtypes = []
                for dtype_combination in dtype_combinations:
                    if type(dtype_combination) == torch.dtype:
                        dtype_combination = (dtype_combination,)
                    dtype_test_name = test_name
                    skipped = False
                    for dtype in dtype_combination:
                        dtype_test_name += '_' + str(dtype).split('.')[1]
                    for dtype in dtype_combination:
                        if dtype in cls.unsupported_dtypes:
                            reason = 'XPU does not support dtype {0}'.format(
                                str(dtype))

                            @wraps(test)
                            def skipped_test(self, *args, reason=reason, **kwargs):
                                raise unittest.SkipTest(reason)

                            assert not hasattr(
                                cls, dtype_test_name), 'Redefinition of test {0}'.format(
                                dtype_test_name)
                            skipped = True
                            setattr(cls, dtype_test_name, skipped_test)
                            break
                        if dtype in [torch.float, torch.double, torch.bfloat16]:
                            floating_precision = DPCPPTestBase._alt_lookup(
                                TORCH_TEST_PRECISIONS,
                                [dtype_test_name, test_name, test.__name__],
                                DEFAULT_FLOATING_PRECISION)
                            if dtype not in test.precision_overrides or test.precision_overrides[
                                    dtype] < floating_precision:
                                test.precision_overrides[dtype] = floating_precision

                    if class_name in disabled_torch_tests and match_name(
                            dtype_test_name, disabled_torch_tests[class_name]):
                        skipped = True
                        setattr(cls, dtype_test_name, disallowed_test)
                    if not skipped:
                        xpu_dtypes.append(
                            dtype_combination
                            if len(dtype_combination) > 1 else dtype_combination[0])
                if len(xpu_dtypes) != 0:
                    test.dtypes[cls.device_type] = xpu_dtypes
                    super().instantiate_test(name, copy.deepcopy(test), generic_cls=generic_cls)


class dtypes(object):
    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all').
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), \
                    "When one dtype variant is a tuple or list, " \
                    "all dtype variants must be. " \
                    "Received non-list non-tuple dtype {0}".format(str(arg))
                assert all(isinstance(dtype, torch.dtype) for dtype in arg), "Unknown dtype in {0}".format(str(arg))
        else:
            assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))

        self.args = args
        self.device_type = kwargs.get('device_type', 'all')

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

class dtypesIfXPU(dtypes):

    def __init__(self, *args):
        super(dtypesIfXPU, self).__init__(*args, device_type="xpu")


class onlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, device, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, device, *args, **kwargs)

        return only_fn

# Overrides specified dtypes on the CPU.

def onlyXPU(fn):
    return onlyOn('xpu')(fn)


def onlyOnCPUAndXPU(fn):
    @wraps(fn)
    def only_fn(self, device, *args, **kwargs):
        if self.device_type != 'cpu' and self.device_type != 'xpu':
            reason = "Doesn't run on {0}".format(self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, device, *args, **kwargs)

    return only_fn


def skipIfXPU(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_XPU:
            raise unittest.SkipTest(
                "test doesn't currently work on the XPU stack")
        else:
            fn(*args, **kwargs)
    return wrapper

def skipXPUIfNoOneMKL(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.xpu.has_onemkl():
            raise unittest.SkipTest(
                "test doesn't currently work without oneMKL")
        else:
            fn(*args, **kwargs)
    return wrapper

class skipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, device, *args, **kwargs)
        return dep_fn


class skipXPUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='xpu')


def _has_sufficient_memory(device, size):
    if torch.device(device).type == 'xpu':
        if not torch.xpu.is_available():
            return False
        gc.collect()
        torch.xpu.empty_cache()
        return torch.xpu.get_device_properties(device).global_memory_size - torch.xpu.memory_allocated(device) >= size
    else:
        raise unittest.SkipTest('Unknown device type')

def largeTensorTestXPU(size, device=None):
    """Skip test if the device has insufficient memory to run the test

    size may be a number of bytes, a string of the form "N GB", or a callable

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overriden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    if isinstance(size, str):
        assert size.endswith("GB") or size.endswith("gb"), "only bytes or GB supported"
        size = 1024 ** 3 * int(size[:-2])

    def inner(fn):
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            _device = device if device is not None else self.get_primary_device()
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest('Insufficient {} memory'.format(_device))

            return fn(self, *args, **kwargs)
        return dep_fn
    return inner

class XPUSyncGuard:
    def __init__(self, sync_debug_mode):
        self.mode = sync_debug_mode

    def __enter__(self):
        # FixMe: torch.xpu.get_/set_sync_debug_mode not implemented
        self.debug_mode_restore = torch.xpu.get_sync_debug_mode()
        torch.xpu_set_sycn_debug_mode(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.xpu.set_sync_debug_mode(self.debug_mode_restore)

TEST_CLASS = DPCPPTestBase
