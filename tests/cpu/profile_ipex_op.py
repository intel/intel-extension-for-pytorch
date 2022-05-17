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

test = torch.tensor([[1.0,1.0],[1.0,1.0]])
model = inplace_softmax().eval()
with torch.no_grad():
    model = torch.jit.trace(model, test)
for _ in range(10):
        res = model(test)

# UT is testing the usecases mentioned in PyTorch doc
# https://pytorch.org/docs/stable/profiler.html

# usecase 1
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof1:
    for _ in range(10):
        res = model(test)
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
        res = model(test)
        prof2.step()


