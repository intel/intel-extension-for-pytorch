import torch
import intel_pytorch_extension as ipex
from torch.utils import ThroughputBenchmark
import argparse

class Interaction(torch.nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()

    def forward(self, x):
        return ipex.interaction(*x)

def inference_benchmark(num_instance, interact_module, dtype, result_dir):
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
    result = open(result_dir, 'a+')
    result.writelines("*" * 50 + "\n")
    result.writelines("dtype=%s, inference"%(dtype) + "\n")
    result.writelines(stats.__str__())
    result.write("\n")
    result.close()
    print(stats)

def training_benchmark(interact_module, dtype, result_dir):
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
    result = open(result_dir, 'a+')
    result.writelines("*" * 50 + "\n")
    result.writelines("dtype=%s, training"%(dtype) + "\n")
    result.write("Took {} ms in average to run {} FW+BW".format(avg_elapsed, "interaction"))
    result.write("\n")
    result.close()
    print("Took {} ms in average to run {} FW+BW".format(avg_elapsed, "interaction"))



def run():
    parser = argparse.ArgumentParser(
        description="benchmark for ipex interaction"
    )
    parser.add_argument("--num-instance", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--result-dir", type=str, default="./logs/interaction-bench-onednn.log")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    interact_module = Interaction()
    if args.inference:
        inference_benchmark(args.num_instance, interact_module, dtype, args.result_dir)
    else:
        training_benchmark(interact_module, dtype, args.result_dir)

if __name__ == "__main__":
    run()
