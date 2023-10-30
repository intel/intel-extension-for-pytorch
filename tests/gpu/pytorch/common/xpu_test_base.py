import __main__
import intel_extension_for_pytorch
import torch
import time

from torch.testing._internal.common_device_type import (
    CUDATestBase,
    filter_desired_device_types,
    PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY,
    PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY,
    _update_param_kwargs,
    _dtype_test_suffix)
from torch.testing._internal.common_utils import compose_parametrize_fns, discover_test_cases_recursively

import subprocess
import os
import sys
import copy
import inspect
import unittest
from functools import wraps
from collections import OrderedDict

cur_script_path = os.path.dirname(os.path.abspath(__file__))
test_suite_root = os.path.join(cur_script_path, "../")
# test_suite_path = os.path.join(test_suite_root, "test/")

sys.path.append(test_suite_root)
# sys.path.append(test_suite_path)

from tool.file_utils import load_from_yaml, save_to_yaml
from tool.case_utils import match_name, match_dtype


static_skipped_cases_list = []
static_skipped_dicts = load_from_yaml(
    os.path.join(test_suite_root, "config/static_skipped_cases_list.yaml"))
if static_skipped_dicts:
    for static_skipped_dict in static_skipped_dicts:
        static_skipped_cases_list.extend(static_skipped_dict['cases'])

USE_DYNAMIC_SKIP = True
dynamic_skipped_cases_list = []
dynamic_skipped_dicts = load_from_yaml(
    os.path.join(test_suite_root, "config/dynamic_skipped_cases_list.yaml"))
if dynamic_skipped_dicts:
    for dynamic_skipped_dict in dynamic_skipped_dicts:
        dynamic_skipped_cases_list.extend(dynamic_skipped_dict['cases'])

unsupported_dtypes = [
    torch.complex,
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.chalf,
    torch.cfloat,
    torch.cdouble,
]

def reload_dyn_skip_list():
    global dynamic_skipped_cases_list
    dynamic_skipped_cases_list = []
    dynamic_skipped_dicts = load_from_yaml(
        os.path.join(test_suite_root, "config/dynamic_skipped_cases_list.yaml"))
    if dynamic_skipped_dicts:
        for dynamic_skipped_dict in dynamic_skipped_dicts:
            dynamic_skipped_cases_list.extend(dynamic_skipped_dict['cases'])

def customized_skipper():
    start_config_time = time.perf_counter()
    # load dynamic skipped cases list every time we start a run
    reload_dyn_skip_list()
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    test_cases = discover_test_cases_recursively(suite)
    for test in test_cases:
        # static skip. Won't run these cases anytime.
        @wraps(test)
        def disallowed_test(self, *args, **kwargs):
            raise unittest.SkipTest("shouldn't run on XPU")
            return test(self, *args, **kwargs)

        # dynamic skip. Should be fixed and re-checked if necessary
        @wraps(test)
        def unsupported_test(self, *args, **kwargs):
            raise unittest.SkipTest("not ready on XPU")
            return test(self, *args, **kwargs)

        @wraps(test)
        def unsupported_dtype_test(self, *args, **kwargs):
            raise unittest.SkipTest("dtype not implement on XPU")
            return test(self, *args, **kwargs)

        test_case_full_name = test.id().split('.', 1)[1][::-1].replace('.', "::", 1)[::-1]
        case_name = test_case_full_name.split("::", 1)[1]
        test_class = test.__class__
        if match_name(test_case_full_name, static_skipped_cases_list):
            setattr(test_class, case_name, disallowed_test)
            print(f"[INFO] trying to skip static case {test_case_full_name}")
        elif match_name(test_case_full_name, dynamic_skipped_cases_list) and USE_DYNAMIC_SKIP:
            setattr(test_class, case_name, unsupported_test)
            print(f"[INFO] trying to skip dynamic case {test_case_full_name}")
        elif match_dtype(test_case_full_name, unsupported_dtypes):
            setattr(test_class, case_name, unsupported_dtype_test)
            print(f"[INFO] trying to skip unsupported dtype case {test_case_full_name}")
    end_config_time = time.perf_counter()
    config_duration = end_config_time - start_config_time
    print(f"[INFO] the configuration time is {config_duration}")
# Test Base is which used to generate each device type specific test class instances
class XPUTestBase(CUDATestBase):
    # we must cheat PyTorch to run XPU cases as what CUDA runs to change this field in case to "cuda"
    device_type = "cuda"  

    # override the DeviceTypeTestBase's instantiate_test to:
    #   1. enable skip mechanism for XPU test base
    #   2. generate each case with suffix 'xpu' but cheat the script to run with 'cuda' which will be converted by tool
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        # safely copied code from pytorch/torch/testing/_internal/common_device_type.py
        # all below are run with device_type as 'cuda'
        def instantiate_test_helper(cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []):
            # Add the device param kwarg if the test needs device or devices.
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if 'device' in test_sig_params or 'devices' in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, 'num_required_devices'):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, 'device', device_arg)

            # Apply decorators based on param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    # check if rte should stop entire test suite.
                    self._stop_test_suite = self._should_stop_test_suite()
                    # Check if test has been decorated with `@expectedFailure`
                    # Using `__unittest_expecting_failure__` attribute, see
                    # https://github.com/python/cpython/blob/ffa505b580464/Lib/unittest/case.py#L164
                    # In that case, make it fail with "unexpected success" by suppressing exception
                    if getattr(test, "__unittest_expecting_failure__", False) and self._stop_test_suite:
                        import sys
                        print("Suppressing fatal exception to trigger unexpected success", file=sys.stderr)
                        return
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), "Redefinition of test {0}".format(name)
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            # By default, no parametrization is needed.
            yield (test, '', {}, lambda _: [])

        # Parametrization decorators set the parametrize_fn attribute on the test.
        parametrize_fn = test.parametrize_fn if hasattr(test, 'parametrize_fn') else default_parametrize_fn

        # If one of the @dtypes* decorators is present, also parametrize over the dtypes set by it.
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                for dtype in dtypes:
                    param_kwargs: Dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, "dtype", dtype)

                    # Note that an empty test suffix is set here so that the dtype can be appended
                    # later after the device.
                    yield (test, '', param_kwargs, lambda _: [])

            parametrize_fn = compose_parametrize_fns(dtype_parametrize_fn, parametrize_fn)

        # Instantiate the parametrized tests.
        for (test, test_suffix, param_kwargs, decorator_fn) in parametrize_fn(test, generic_cls, cls):
            test_suffix = '' if test_suffix == '' else '_' + test_suffix
            device_suffix = '_xpu'  # this line was changed to 'xpu' specifically

            # Note: device and dtype suffix placement
            # Special handling here to place dtype(s) after device according to test name convention.
            dtype_kwarg = None
            if 'dtype' in param_kwargs or 'dtypes' in param_kwargs:
                dtype_kwarg = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else param_kwargs['dtype']
            test_name = '{}{}{}{}'.format(name, test_suffix, device_suffix, _dtype_test_suffix(dtype_kwarg))

            instantiate_test_helper(cls=cls, name=test_name, test=test, param_kwargs=param_kwargs,
                                    decorator_fn=decorator_fn)


# Re-define instantiate_device_type_tests to override PyTorch's one.
# This makes us have the ability to:
#   1. only run XPU cases without change the ported code
#   2. generate test class with 'XPU' suffix together with passing device_type as 'cuda' to PyTorch*
#   3. call XPUTestBase.instantiate_test without copying too much code outside from PyTorch*
# The following part should be kept the same as possible as which in pytorch/torch/testing/_internal/common_device_type.py
# with only change a few lines to meet our requirements
def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
    del scope[generic_test_class.__name__]
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]
    # cheat device_type as 'cuda'
    XPUTestBase.device_type = "cuda"
    test_bases = [XPUTestBase]
    desired_device_type_test_bases = filter_desired_device_types(test_bases, except_for, only_for)

    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))
    
    # for env users which knows they are running xpu test, the device_type should be temp changed to 'xpu'
    XPUTestBase.device_type = "xpu"
    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
                                                                 env_except_for, env_only_for)

    for base in desired_device_type_test_bases:
        class_name = generic_test_class.__name__ + base.device_type.upper()     # 'XPU' here
        XPUTestBase.device_type = "cuda"    # keep cheating the test suite
        device_type_test_class: Any = type(class_name, (base, empty_class), {})
        for name in generic_members:
            if name in generic_tests:
                test = getattr(generic_test_class, name)
                device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
            else:
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


# override module method with XPU's one defined above
try:
    mod = sys.modules['torch.testing._internal.common_device_type']
    mod.instantiate_device_type_tests = instantiate_device_type_tests
    print("[INFO] Override instantiate_device_type_tests in torch.testing._internal.common_device_type with XPU's one")
except KeyError as e_key:
    print(e_key, file=sys.stderr)
    print("[ERROR] Failed to override instantiate_device_type_tests in torch.testing._internal.common_device_type", file=sys.stderr)
    exit(1)

# Workaround to avoid cuda checking its versions
try:
    import torch.testing._internal.common_cuda
    mod = sys.modules['torch.testing._internal.common_cuda']
    mod.tf32_is_not_fp32 = lambda: True
    print("[INFO] Override tf32_is_not_fp32 in torch.testing._internal.common_cuda with lambda always returns True")
except KeyError as e_key:
    print(e_key, file=sys.stderr)
    print("[ERROR] Failed to override tf32_is_not_fp32 in torch.testing._internal.common_cuda", file=sys.stderr)
    exit(1)

# Workaround to avoid cuda checking its versions
try:
    import torch.testing._internal.jit_utils
    mod = sys.modules['torch.testing._internal.jit_utils']
    mod.RUN_CUDA = torch.xpu.is_available()
    mod.RUN_CUDA_MULTI_GPU = mod.RUN_CUDA and torch.xpu.device_count() > 1
    mod.RUN_CUDA_HALF = mod.RUN_CUDA
    print("[INFO] Pre-load torch.testing._internal.jit_utils to bypass some global check")
except KeyError as e_key:
    print(e_key, file=sys.stderr)
    print("[ERROR] Failed to pre-load torch.testing._internal.jit_utils", file=sys.stderr)
    exit(1)

# Workaround to avoid cuda checking its versions
try:
    import torch.testing._internal.inductor_utils
    mod = sys.modules['torch.testing._internal.inductor_utils']
    from intel_extension_for_pytorch._inductor.xpu import utils
    mod.HAS_CUDA = utils.has_triton()
    print("[INFO] Pre-load torch.testing._internal.inductor_utils to bypass some global check")
    import torch._inductor.utils
    mod = sys.modules['torch._inductor.utils']
    mod.has_triton = utils.has_triton
    print("[INFO] Pre-load torch._inductor.utils to bypass some global check")
except KeyError as e_key:
    print(e_key, file=sys.stderr)
    print("[ERROR] Failed to pre-load torch.testing._internal.inductor_utils", file=sys.stderr)
    print("[ERROR] Failed to pre-load torch._inductor.utils", file=sys.stderr)
    exit(1)

# Workaround to avoid involving nvfuser by pre-load torch._prim
try:
    import torch._prims
    import torch._prims.context
    import torch._prims.executor
    #import torch._prims.nvfuser_executor
    import torch._prims_common
    print("[INFO] Pre-load torch._prims and all things under this module without changing 'cuda' to 'xpu'")
except ImportError as e_import:
    print(e_import, file=sys.stderr)
    print("[ERROR] Failed to pre-load torch._prims and things under this module", file=sys.stderr)
    exit(1)

# Try to import model convert tool
try:
    import model_convert
except ImportError as e_import:
    print(e_import, file=sys.stderr)
    print("[ERROR] Run PyTorch Ported UT must install scripts/tools/model_convert tool first.", file=sys.stderr)
    exit(1)

# Temp workaround to solve those not supported cuda apis
# Wait for fixing from grangye and remove following lines
torch.cuda.is_bf16_supported = lambda: True 
torch.cuda.is_initialized = lambda: True
torch.cuda.get_device_capability = lambda x=None: (8, 6)
