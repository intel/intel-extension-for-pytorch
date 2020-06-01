import intel_pytorch_extension as ipex

class AutoMixPrecision(object):
    def __init__(self, enable_or_not = False):
        self.old_value = ipex.core.get_mix_bf16_fp32()
        self.enable_or_not = enable_or_not

    def __enter__(self):
        if self.enable_or_not:
            ipex.core.enable_mix_bf16_fp32()
        else:
            ipex.core.disable_mix_bf16_fp32()

    def __exit__(self, *args, **kwargs):
        if self.old_value:
            ipex.core.enable_mix_bf16_fp32()
        else:
            ipex.core.disable_mix_bf16_fp32()

class AutoDNNL(object):
    def __init__(self, enable_or_not = False):
        self.old_value = ipex.core.get_auto_dnnl()
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
