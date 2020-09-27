import torch
import intel_pytorch_extension as ipex

class AutoMixPrecision(object):
    def __init__(self, enable_or_not = False, train = False):
        self.old_value = ipex.get_auto_mix_precision()
        self.pre_running_mode = 'training' if ipex.get_train() else 'inference'
        self.enable_or_not = enable_or_not
        self.running_mode = 'training' if train else 'inference'

    def __enter__(self):
        if self.enable_or_not:
            ipex.enable_auto_mix_precision(ipex.AmpConf(torch.bfloat16), self.running_mode).__enter__()
        else:
            ipex.enable_auto_mix_precision(ipex.AmpConf(None)).__enter__()

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.enable_auto_mix_precision(ipex.AmpConf(torch.bfloat16), self.pre_running_mode).__enter__()
        else:
            ipex.enable_auto_mix_precision(ipex.AmpConf(None)).__enter__()

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
