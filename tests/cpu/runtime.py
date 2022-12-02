import argparse
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

def create_cpu_pool(args):
    core_ids = [1, 2]
    cpu_pool = ipex.cpu.runtime.CPUPool(core_ids)
    print("The created CPUPool has core is: {}".format(cpu_pool.core_ids), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default="create_cpu_pool", type=str)
    args = parser.parse_args()
    if args.case_name == "create_cpu_pool":
        create_cpu_pool(args)
