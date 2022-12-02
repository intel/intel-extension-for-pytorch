import argparse
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

class inplace_softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x + 1
        x2 = nn.Softmax(dim=-1)(x1)
        return x2

def run_profile(model, x):
    with torch.no_grad():
        model = torch.jit.trace(model, x)
    for _ in range(10):
            res = model(x)

    # UT is testing the usecases mentioned in PyTorch doc
    # https://pytorch.org/docs/stable/profiler.html

    # usecase 1
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof1:
        for _ in range(10):
            res = model(x)
    print(prof1.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

    # usecase 2
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
            wait=2,
            warmup=3,
            active=5),
            on_trace_ready=trace_handler
        ) as prof2:
        for _ in range(10):
            res = model(x)
            prof2.step()

def run_model(llga):
    llga_enabled = ipex._C.is_llga_fp32_bf16_enabled()
    if llga:
        ipex._C.set_llga_fp32_bf16_enabled(True)    
    
    x = torch.tensor([[1.0,1.0],[1.0,1.0]])
    model = inplace_softmax().eval()    
    
    run_profile(model, x)
    
    ipex._C.set_llga_fp32_bf16_enabled(llga_enabled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llga", action='store_true', help='use llga for fp32 path', default=False)
    args = parser.parse_args()
    run_model(args.llga)