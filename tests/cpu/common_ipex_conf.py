import torch
import intel_pytorch_extension as ipex

class AutoMixPrecision(object):
    def __init__(self, enable_or_not = False, train = False):
        self.old_value = ipex.get_auto_mix_precision()
        self.train_old_value = ipex.get_train()
        self.enable_or_not = enable_or_not
        self.train = train

    def __enter__(self):
        if self.enable_or_not:
            ipex.enable_auto_mix_precision(mixed_dtype=torch.bfloat16, train=self.train)
        else:
            ipex.enable_auto_mix_precision(mixed_dtype=None)

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.enable_auto_mix_precision(mixed_dtype=torch.bfloat16, train=self.train_old_value)
        else:
            ipex.enable_auto_mix_precision(mixed_dtype=None)

class AutoDNNL(object):
    def __init__(self, enable_or_not = False):
        self.old_value = ipex.get_auto_optimization()
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
