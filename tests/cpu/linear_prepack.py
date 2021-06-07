import torch
import intel_pytorch_extension as ipex
from common_utils import int8_calibration

ipex.core.enable_auto_dnnl()

ic = 1024
oc = 1024
bs = 16

LL = torch.nn.Linear(ic, oc).to(ipex.DEVICE)

def get_input():
    return torch.rand(bs, ic).to(ipex.DEVICE)

def run_linear(auto_mix_conf=None):
    for i in range(3):
        if auto_mix_conf != None:
            with ipex.AutoMixPrecision(auto_mix_conf):
                LL(get_input())
        else:
            LL(get_input())

if __name__ == "__main__":
    print(f"fp32, {'*' * 50}") 
    run_linear()

    print(f"auto-mix for bf16, {'*' * 50}") 
    bf16_conf = ipex.AmpConf(torch.bfloat16)
    run_linear(bf16_conf)

    print(f"back to fp32, {'*' * 50}") 
    ipex.core.reorder_to_float32(LL.weight)
    ipex.core.reorder_to_float32(LL.bias)
    run_linear()

    print(f"auto-mix for int8, {'*' * 50}") 
    int8_calibration(LL,  [get_input() for i in range(3)], "./int8.config")
    int8_conf = ipex.AmpConf(torch.int8, "./int8.config")
    run_linear(int8_conf)

    print(f"back to fp32, {'*' * 50}") 
    ipex.core.reorder_to_float32(LL.weight)
    ipex.core.reorder_to_float32(LL.bias)
    run_linear()