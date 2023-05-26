import torch
import intel_extension_for_pytorch as ipex
from functools import wraps


class AutoMixPrecision(object):
    def __init__(self, enable_or_not=False, train=False):
        self.old_value = ipex.get_auto_mix_precision()
        self.train_old_value = ipex.get_train()
        self.enable_or_not = enable_or_not
        self.train = train

    def __enter__(self):
        if self.enable_or_not:
            ipex.enable_auto_mixed_precision(
                mixed_dtype=torch.bfloat16, train=self.train
            )
        else:
            ipex.enable_auto_mixed_precision(mixed_dtype=None)

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.enable_auto_mixed_precision(
                mixed_dtype=torch.bfloat16, train=self.train_old_value
            )
        else:
            ipex.enable_auto_mixed_precision(mixed_dtype=None)


class AutoDNNL(object):
    def __init__(self, enable_or_not=False):
        self.old_value = ipex._get_auto_optimization()
        self.enable_or_not = enable_or_not

    def __enter__(self):
        if self.enable_or_not:
            ipex.core.enable_auto_dnnl()
        else:
            ipex.core.disable_auto_dnnl()

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.core.enable_auto_dnnl()
        else:
            ipex.core.disable_auto_dnnl()


def runtime_thread_affinity_test_env(func):
    @wraps(func)
    def wrapTheFunction(*args):
        # In some cases, the affinity of main thread may be changed: MultiStreamModule of stream 1
        # Ensure, we restore the affinity of main thread
        previous_cpu_pool = ipex._C.get_current_cpu_pool()
        func(*args)
        ipex._C.set_cpu_pool(previous_cpu_pool)

    return wrapTheFunction
