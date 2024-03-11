import sys
import argparse
from functools import partial
from .utils._logger import logger, WarningType

from .cpu.launch import (
    init_parser as cpu_init_parser,
    run_main_with_args as cpu_run_main_with_args,
)
from .xpu.launch import (
    init_parser as xpu_init_parser,
    run_main_with_args as xpu_run_main_with_args,
)


def init_parser():
    parser = argparse.ArgumentParser(
        description="\n=================================== LAUNCHER ============================== \n"
        "\nThis is a script for launching PyTorch training and inference on *Intel Xeon CPU* "
        "or *Intel GPU* with optimal configurations. \n"
        "\n################################# Basic usage ############################# \n"
        "\n1. Run with CPU backend \n"
        "\n    >>> ipexrun cpu python_script args\n"
        "\n2. Run with XPU backend\n"
        "\n    >>> ipexrun xpu python_script args\n"
        "\n############################################################################# \n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="backend", help="Run with specific Backend")

    cpu_parser = subparsers.add_parser(
        "cpu",
        help="Run with CPU Backend",
        description="This is a script for launching PyTorch training and inference on Intel Xeon CPU "
        "with optimal configurations. Now, single instance inference/training, multi-instance "
        "inference/training and distributed training with oneCCL backend is enabled. "
        "To get the peak performance on Intel Xeon CPU, the script optimizes the configuration "
        "of thread and memory management. For thread management, the script configures thread "
        "affinity and the preload of Intel OMP library. For memory management, it configures "
        "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
        "\n################################# Basic usage ############################# \n"
        "\n 1. single instance\n"
        "\n   >>> ipexrun cpu python_script args \n"
        "\n2. multi-instance \n"
        "\n    >>> ipexrun cpu --ninstances xxx --ncore_per_instance xx python_script args\n"
        "\n3. Single-Node multi-process distributed training\n"
        "\n    >>> python  -m intel_extension_for_pytorch.launch cpu --distributed  python_script args\n"
        "\n4. Multi-Node multi-process distributed training: (e.g. two nodes)\n"
        "\n   rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*\n"
        "\n   >>> ipexrun cpu --distributed --nproc_per_node=2\n"
        "\n       --nnodes=2 --hostfile hostfile python_script args\n"
        "\n############################################################################# \n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    xpu_parser = subparsers.add_parser(
        "xpu",
        help="Run with XPU Backend",
        description="This is a script for launching PyTorch training and inference on Intel GPU Series "
        "with optimal configurations."
        "\n################################# Basic usage ############################# \n"
        "\n single instance\n"
        "\n   >>> ipexrun xpu python_script args \n"
        "\n############################################################################# \n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    cpu_init_parser(cpu_parser)
    xpu_init_parser(xpu_parser)
    return parser, cpu_parser, xpu_parser


def mixed_print_help(f1, f2, f3):
    f1()
    print(
        "\n================================ CPU LAUNCHER ============================= \n"
    )
    f2()
    print(
        "\n================================ XPU LAUNCHER ============================= \n"
    )
    f3()


def main():
    parser, cpu_parser, xpu_parser = init_parser()
    origin_parser_print_help = parser.print_help
    cpu_parser_print_help = cpu_parser.print_help
    xpu_parser_print_help = xpu_parser.print_help
    parser.print_help = partial(
        mixed_print_help,
        f1=origin_parser_print_help,
        f2=cpu_parser_print_help,
        f3=xpu_parser_print_help,
    )
    if (
        len(sys.argv) > 1
        and "-h" not in sys.argv
        and "--help" not in sys.argv
        and sys.argv[1] != "cpu"
        and sys.argv[1] != "xpu"
    ):
        msg = (
            "Backend is not specified, it will automatically default to cpu."
            + "Please start with ipexrun <cpu or xpu> python_script args"
        )
        logger.warning(msg, _type=WarningType.MissingArgument)
        sys.argv.insert(1, "cpu")
    args = parser.parse_args()
    if args.backend == "cpu":
        cpu_run_main_with_args(args)
    elif args.backend == "xpu":
        xpu_run_main_with_args(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
