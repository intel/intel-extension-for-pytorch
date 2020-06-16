import intel_pytorch_extension as ipex

class AutoMixPrecision(object):
    def __init__(self, enable_or_not = False):
        self.old_value = ipex.get_auto_mix_precision()
        self.enable_or_not = enable_or_not

    def __enter__(self):
        if self.enable_or_not:
            ipex.enable_auto_mix_precision(bf16=True)
        else:
            ipex.enable_auto_mix_precision(bf16=False)

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.enable_auto_mix_precision(bf16=True)
        else:
            ipex.enable_auto_mix_precision(bf16=False)

class AutoDNNL(object):
    def __init__(self, enable_or_not = False):
        self.old_value = ipex.get_auto_optimization()
        self.enable_or_not = enable_or_not

    def __enter__(self):
        if self.enable_or_not:
            ipex.enable_auto_optimization()
        else:
            ipex.enable_auto_optimization(False)

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.enable_auto_optimization()
        else:
            ipex.enable_auto_optimization(False)
