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
for dtype in ["fp32"]:
    test3 = torch.tensor([[1.0,1.0],[1.0,1.0]])
    if dtype == "bf16":
        test3 = test3.bfloat16()
    model3 = inplace_softmax().eval()
    with torch.no_grad():
        model3 = torch.jit.trace(model3, test3)
    with torch.profiler.profile(
           activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
            wait=2,
            warmup=3,
            active=5),
            on_trace_ready=trace_handler
       ) as prof:
        for _ in range(10):
            res3 = model3(test3)
            prof.step()
