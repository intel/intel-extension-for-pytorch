import platform
import os
import glob
import argparse
from argparse import SUPPRESS, OPTIONAL, ZERO_OR_MORE
import logging
from datetime import datetime
import intel_extension_for_pytorch.cpu.auto_ipex as auto_ipex
from .launcher_distributed import DistributedTrainingLauncher
from .launcher_multi_instances import MultiInstancesLauncher

"""
This is a script for launching PyTorch training and inference on Intel Xeon CPU with optimal configurations.
Now, single instance inference/training, multi-instance inference/training and distributed training
with oneCCL backend is enabled.

To get the peak performance on Intel Xeon CPU, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

**How to use this module:**

*** Single instance inference/training ***

1. Run single-instance inference or training on a single node with all CPU nodes.

::

   >>> ipexrun --throughput-mode script.py args

2. Run single-instance inference or training on a single CPU node.

::

   >>> ipexrun --nodes-list 1 script.py args

*** Multi-instance inference ***

1. Multi-instance
   By default, one instance per node. if you want to set the instance numbers and core per instance,
   --ninstances and  --ncores-per-instance should be set.


   >>> ipexrun python_script args

   eg: on CLX8280 with 14 instance, 4 cores per instance
::

   >>> ipexrun  --ninstances 14 --ncores-per-instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify instance_idx.

   eg: run 0th instance among SKX with 2 instance (i.e., numactl -C 0-27)
::

   >>> ipexrun  --ninstances 2 --instance-idx 0 python_script args

   eg: run 1st instance among SKX with 2 instance (i.e., numactl -C 28-55)
::

   >>> ipexrun  --ninstances 2 --instance-idx 1 python_script args

   eg: run 0th instance among SKX with 2 instance, 2 cores per instance, first four cores (i.e., numactl -C 0-1)
::

   >>> ipexrun  --cores-list "0-3" --ninstances 2 --ncores-per-instance 2 --instance-idx 0 python_script args

*** Distributed Training ***

spawns up multiple distributed training processes on each of the training nodes. For intel_extension_for_pytorch, oneCCL
is used as the communication backend and MPI used to launch multi-proc. To get the better
performance, you should specify the different cores for oneCCL communication and computation
process seperately. This tool can automatically set these ENVs(such as I_MPI_PIN_DOMIN) and launch
multi-proc for you.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned.  It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.


1. Single-Node multi-process distributed training

::

    >>> ipexrun --nnodes N  python_script  --arg1 --arg2 --arg3 and all other
                arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 29500)*

::

    >>> ipexrun --nnodes 2 --nprocs-per-node=xxx
               --hostfile hostfile python_sript --arg1 --arg2 --arg3
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> ipexrun --help

*** Memory allocator  ***

Memory allocator plays an important role from performance perspective as well. A more efficient memory usage reduces
overhead on unnecessary memory allocations or destructions, and thus results in a faster execution. JeMalloc and
TCMalloc can be used as substitution of the default memory allocator. It is as easy as setting the
`--memory-allocator` argument to either of `auto`, `default`, `jemalloc` and `tcmalloc`. Setting it to `auto` tries
searching availability of the memory allocator in order of `tcmalloc`, 'jemalloc` and 'default`.

"""


def add_deprecated_params(parser):
    group = parser.add_argument_group("Deprecated Arguments")
    group.add_argument(
        "--nproc_per_node",
        metavar="\b",
        type=int,
        default=-1,
        help="Deprecated by --nprocs-per-node.",
    )
    group.add_argument(
        "--more_mpi_params",
        metavar="\b",
        type=str,
        default="",
        help="Deprecated by --extra-mpi-params.",
    )
    group.add_argument(
        "--ncore_per_instance",
        metavar="\b",
        type=int,
        default=-1,
        help="Deprecated by --ncores-per-instance.",
    )
    group.add_argument(
        "--node_id",
        metavar="\b",
        type=int,
        default=-1,
        help="Deprecated by --nodes-list.",
    )
    group.add_argument(
        "--core_list",
        metavar="\b",
        type=str,
        default="",
        help="Deprecated by --cores-list.",
    )
    group.add_argument(
        "--logical_core_for_ccl",
        action="store_true",
        default=False,
        help="Deprecated by --logical-cores-for-ccl.",
    )
    group.add_argument(
        "--enable_tcmalloc",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--enable_jemalloc",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--use_default_allocator",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--use_logical_core",
        action="store_true",
        default=False,
        help="Deprecated by --use-logical-cores.",
    )
    group.add_argument(
        "--disable_numactl",
        action="store_true",
        default=False,
        help="Deprecated by --multi-task-manager.",
    )
    group.add_argument(
        "--disable_taskset",
        action="store_true",
        default=False,
        help="Deprecated by --multi-task-manager.",
    )
    group.add_argument(
        "--disable_iomp",
        action="store_true",
        default=False,
        help="Deprecated by --omp-runtime.",
    )
    group.add_argument(
        "--log_path", type=str, default="", help="Deprecated by --log-dir."
    )
    group.add_argument(
        "--multi_instance",
        action="store_true",
        default=False,
        help="Deprecated. Will be removed.",
    )
    group.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Deprecated. Will be removed.",
    )


def process_deprecated_params(args, logger):
    if args.nproc_per_node != -1:
        logger.warning("Argument --nproc_per_node is deprecated by --nprocs-per-node.")
        args.nprocs_per_node = args.nproc_per_node
    if args.more_mpi_params != "":
        logger.warning(
            "Argument --more_mpi_params is deprecated by --extra-mpi-params."
        )
        args.extra_mpi_params = args.more_mpi_params
    if args.ncore_per_instance != -1:
        logger.warning(
            "Argument --ncore_per_instance is deprecated by --ncores-per-instance."
        )
        args.ncores_per_instance = args.ncore_per_instance
    if args.node_id != -1:
        logger.warning("Argument --node_id is deprecated by --nodes-list.")
        args.nodes_list = str(args.node_id)
    if args.core_list != "":
        logger.warning("Argument --core_list is deprecated by --cores-list.")
        args.cores_list = args.core_list
    if args.logical_core_for_ccl:
        logger.warning(
            "Argument --logical_core_for_ccl is deprecated by --logical-cores-for-ccl."
        )
        args.logical_cores_for_ccl = args.logical_core_for_ccl
    if args.use_logical_core:
        logger.warning(
            "Argument --use_logical_core is deprecated by --use-logical-cores."
        )
        args.use_logical_cores = args.use_logical_core
    if args.log_path != "":
        logger.warning("Argument --log_path is deprecated by --log-dir.")
        args.log_dir = args.log_path

    if args.multi_instance:
        logger.info(
            "Argument --multi_instance is deprecated. Will be removed. \
                If you are using the deprecated argument, please update it to the new one."
        )
    if args.distributed:
        logger.info(
            "Argument --distributed is deprecated. Will be removed. \
                If you are using the deprecated argument, please update it to the new one."
        )

    if args.enable_tcmalloc or args.enable_jemalloc or args.use_default_allocator:
        logger.warning(
            "Arguments --enable_tcmalloc, --enable_jemalloc and --use_default_allocator \
                are deprecated by --memory-allocator."
        )
        if args.use_default_allocator:
            args.memory_allocator = "default"
        if args.enable_jemalloc:
            args.memory_allocator = "jemalloc"
        if args.enable_tcmalloc:
            args.memory_allocator = "tcmalloc"
    if args.disable_numactl:
        logger.warning(
            "Argument --disable_numactl is deprecated by --multi-task-manager."
        )
        args.multi_task_manager = "taskset"
    if args.disable_taskset:
        logger.warning(
            "Argument --disable_taskset is deprecated by --multi-task-manager."
        )
        args.multi_task_manager = "numactl"
    if args.disable_iomp:
        logger.warning("Argument --disable_iomp is deprecated by --omp-runtime.")
        args.omp_runtime = "default"


class ArgumentTypesDefaultsHelpFormatter(argparse.HelpFormatter):
    """Help message formatter which adds default values to argument help.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _fill_text(self, text, width, indent):
        return "".join(indent + line for line in text.splitlines(keepends=True))

    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        help = action.help
        if "%(type)" not in action.help:
            if action.type is not SUPPRESS:
                typeing_nargs = [OPTIONAL, ZERO_OR_MORE]
                if action.option_strings or action.nargs in typeing_nargs:
                    help += " (type: %(type)s)"
        if "%(default)" not in action.help:
            if action.default is not SUPPRESS:
                defaulting_nargs = [OPTIONAL, ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += " (default: %(default)s)"
        return help


def init_parser(parser):
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """

    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        'as a python module, executing with the same behavior as "python -m".',
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        default=False,
        action="store_true",
        help="Avoid applying python to execute program.",
    )

    parser.add_argument(
        "--log-dir",
        "--log_dir",
        default="",
        type=str,
        help="The log file directory. Setting it to empty disables logging to files.",
    )
    parser.add_argument(
        "--log-file-prefix",
        "--log_file_prefix",
        default="run",
        type=str,
        help="log file name prefix",
    )

    # positional
    parser.add_argument(
        "program",
        type=str,
        help="Full path to the proram/script to be launched. "
        "followed by all the arguments for the script",
    )

    # rest from the training program
    parser.add_argument(
        "program_args",
        nargs=argparse.REMAINDER,
    )

    launcher_distributed = DistributedTrainingLauncher()
    launcher_multi_instances = MultiInstancesLauncher()
    launcher_multi_instances.add_common_params(parser)
    launcher_multi_instances.add_params(parser)
    launcher_distributed.add_params(parser)
    auto_ipex.add_auto_ipex_params(parser)
    add_deprecated_params(parser)

    return parser


def run_main_with_args(args):
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_str)
    logger = logging.getLogger(__name__)

    launcher_distributed = DistributedTrainingLauncher(logger)
    launcher_multi_instances = MultiInstancesLauncher(logger)

    process_deprecated_params(args, logger)
    if args.log_dir:
        path = os.path.dirname(
            args.log_dir if args.log_dir.endswith("/") else f"{args.log_dir}/"
        )
        if not os.path.exists(path):
            os.makedirs(path)
        args.log_dir = path

        args.log_file_prefix = (
            f'{args.log_file_prefix}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        )
        fileHandler = logging.FileHandler(
            f"{args.log_dir}/{args.log_file_prefix}_instances.log"
        )
        logFormatter = logging.Formatter(format_str)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    assert args.no_python or args.program.endswith(
        ".py"
    ), 'For non Python script, you should use "--no-python" parameter.'

    env_before = set(os.environ.keys())
    # Verify LD_PRELOAD
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            if item != "":
                matches = glob.glob(item)
                if len(matches) > 0:
                    lst_valid.append(item)
                else:
                    logger.warning(
                        f"{item} doesn't exist. Removing it from LD_PRELOAD."
                    )
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    launcher = None
    if args.nnodes > 0:
        launcher = launcher_distributed
    else:
        launcher = launcher_multi_instances

    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug(f"{x}={os.environ[x]}")
