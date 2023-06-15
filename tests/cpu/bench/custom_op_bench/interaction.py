import torch
import intel_extension_for_pytorch as ipex
from torch.utils import ThroughputBenchmark
import argparse

class Interaction(torch.nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()

    def forward(self, x):
        return ipex.nn.functional.interaction(*x)

def inference_benchmark(num_instance, interact_module, dtype):
    inputs = []
    for i in range(0, 27):
        inputs.append(torch.randn([128, 128]).to(dtype))
    with torch.no_grad():
        interact_module = torch.jit.trace(interact_module, [inputs], check_trace=False)
        bench = ThroughputBenchmark(interact_module)
        bench.add_input(inputs)
        stats = bench.benchmark(
            num_calling_threads=num_instance,
            num_warmup_iters=100,
            num_iters=1000 * num_instance,
        )
    print(stats)

def training_benchmark(interact_module, dtype):
    import time
    inputs = []
    for i in range(0, 27):
        inputs.append(torch.randn([4096, 128]).to(dtype).requires_grad_())
    # warmup
    for _ in range(100):
        y = interact_module.forward(inputs).sum()
        y.backward()

    startT = time.time()
    for _ in range(1000):
        y = interact_module.forward(inputs).sum()
        y.backward()
    endT = time.time()
    avg_elapsed = (endT - startT)
    print("Took {} ms on average to run {} FW+BW".format(avg_elapsed, "interaction"))



def run():
    parser = argparse.ArgumentParser(
        description="benchmark for ipex interaction"
    )
    parser.add_argument("--num-instance", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    interact_module = Interaction()
    if args.inference:
        inference_benchmark(args.num_instance, interact_module, dtype)
    else:
        training_benchmark(interact_module, dtype)

if __name__ == "__main__":
    run()
