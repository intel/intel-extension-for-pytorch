import argparse
from .launch import init_parser, run_main_with_args, ArgumentTypesDefaultsHelpFormatter


def main():
    description = """
This is a script for launching PyTorch training and inference on Intel Xeon CPU with optimal configurations. \
    Now, single instance inference/training, multi-instance inference/training and distributed training with \
    oneCCL backend is enabled.  To get the peak performance on Intel Xeon CPU, the script optimizes the \
    configuration of thread and memory management. For thread management, the script configures thread affinity \
    and the preload of Intel OMP library. For memory management, it configures NUMA binding and preload optimized \
    memory allocation library (e.g. tcmalloc, jemalloc)
################################# Basic usage #############################
1. single instance
   >>> ipexrun python_script args
2. multi-instance
   >>> ipexrun --ninstances N --ncores-per-instance M python_script args
3. Single-Node multi-process distributed training
   >>> ipexrun --nnodes 1 python_script args
4. Multi-Node multi-process distributed training: (e.g. two nodes)
   rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*
   >>> ipexrun --nnodes 2 --nproc-per-node 2
       --hostfile hostfile python_script args
###########################################################################
"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=ArgumentTypesDefaultsHelpFormatter
    )
    parser = init_parser(parser)
    args = parser.parse_args()
    run_main_with_args(args)


if __name__ == "__main__":
    main()
