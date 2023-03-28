from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import platform
import subprocess
import os
from os.path import expanduser
import re
import glob
import numpy as np
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter
import logging
import psutil
from datetime import datetime
import intel_extension_for_pytorch.cpu.auto_ipex as auto_ipex

format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)

r"""
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

   >>> ipexrun --throughput_mode script.py args

2. Run single-instance inference or training on a single CPU node.

::

   >>> ipexrun --node_id 1 script.py args

*** Multi-instance inference ***

1. Multi-instance
   By default, one instance per node. if you want to set the instance numbers and core per instance,
   --ninstances and  --ncore_per_instance should be set.


   >>> ipexrun -- python_script args

   eg: on CLX8280 with 14 instance, 4 cores per instance
::

   >>> ipexrun  --ninstances 14 --ncore_per_instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify instance_idx.

   eg: run 0th instance among SKX with 2 instance (i.e., numactl -C 0-27)
::

   >>> ipexrun  --ninstances 2 --instance_idx 0 python_script args

   eg: run 1st instance among SKX with 2 instance (i.e., numactl -C 28-55)
::

   >>> ipexrun  --ninstances 2 --instance_idx 1 python_script args

   eg: run 0th instance among SKX with 2 instance, 2 cores per instance, first four cores (i.e., numactl -C 0-1)
::

   >>> ipexrun  --core_list "0-3" --ninstances 2 --ncore_per_instance 2 --instance_idx 0 python_script args

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

    >>> python  -m intel_extension_for_pytorch.cpu.launch --distributed  python_script  --arg1 --arg2 --arg3 and all other
                arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 29500)*

::

    >>> ipexrun --distributed --nproc_per_node=xxx
               --nnodes=2 --hostfile hostfile python_sript --arg1 --arg2 --arg3
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> ipexrun --help

*** Memory allocator  ***

"--enable_tcmalloc" and "--enable_jemalloc" can be used to enable different memory allcator.

"""

class CPUinfo():
    '''
    Get CPU inforamation, such as cores list and NUMA information.
    '''
    def __init__(self):

        self.cpuinfo = []
        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
            env_lang = os.getenv('LANG', 'UNSET')
            os.environ['LANG'] = 'C'
            lscpu_info = subprocess.check_output(args, env=os.environ, universal_newlines=True).split("\n")
            if env_lang == 'UNSET':
                del os.environ['LANG']
            else:
                os.environ['LANG'] = env_lang

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))
            assert len(self.cpuinfo) > 0, "cpuinfo is empty"
            self.get_socket_info()

    def get_socket_info(self):
        idx_active = 3
        if self.cpuinfo[0][idx_active] == '':
            idx_active = 2
        self.nodes = int(max([line[idx_active] for line in self.cpuinfo])) + 1
        self.node_physical_cores = []  # node_id is index
        self.node_logical_cores = []   # node_id is index
        self.physical_core_node_map = {}  # phyical core to numa node id
        self.logical_core_node_map = {}   # logical core to numa node id
        self.physical_to_logical = {}

        for node_id in range(self.nodes):
            cur_node_physical_core = []
            cur_node_logical_core = []
            for line in self.cpuinfo:
                nid = line[idx_active] if line[idx_active] != '' else '0'
                if node_id == int(nid):
                    if int(line[1]) not in cur_node_physical_core:
                        cur_node_physical_core.append(int(line[1]))
                        self.physical_core_node_map[int(line[1])] = int(node_id)
                    else:
                        self.physical_to_logical[int(line[1])] = int(line[0])
                    cur_node_logical_core.append(int(line[0]))
                    self.logical_core_node_map[int(line[0])] = int(node_id)
            self.node_physical_cores.append(cur_node_physical_core)
            self.node_logical_cores.append(cur_node_logical_core)

    def node_nums(self):
        return self.nodes

    def physical_core_nums(self):
        return len(self.node_physical_cores) * len(self.node_physical_cores[0])

    def logical_core_nums(self):
        return len(self.node_logical_cores) * len(self.node_logical_cores[0])

    def get_node_physical_cores(self, node_id):
        if node_id < 0 or node_id > self.nodes - 1:
            logger.error("Invalid node id")
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        if node_id < 0 or node_id > self.nodes - 1:
            logger.error("Invalid node id")
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        return np.array(self.node_physical_cores).flatten().tolist()

    def get_all_logical_cores(self):
        return np.array(self.node_logical_cores).flatten().tolist()

    def numa_aware_check(self, core_list):
        '''
        Check whether all cores in core_list are in the same NUMA node. cross NUMA will reduce perforamnce.
        We strongly advice to not use cores on different nodes.
        '''
        cores_numa_map = self.logical_core_node_map
        numa_ids = []
        for core in core_list:
            numa_id = cores_numa_map[core]
            if not numa_id in numa_ids:
                numa_ids.append(numa_id)
        if len(numa_ids) > 1:
            logger.warning("Numa Aware: cores:{} on different NUMA nodes:{}".format(str(core_list), str(numa_ids)))
        if len(numa_ids) == 0:
            logger.error("invalid number of NUMA nodes; please make sure numa_ids >= 1")
            exit(-1)
        return numa_ids

class Launcher():
    r"""
     Base class for launcher
    """
    def __init__(self):
        self.cpuinfo = CPUinfo()

    def launch(self, args):
        pass

    def add_lib_preload(self, lib_type=None):
        '''
        Enable TCMalloc/JeMalloc/intel OpenMP
        '''
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(os.environ["CONDA_PREFIX"] + "/lib/")
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(os.environ["VIRTUAL_ENV"] + "/lib/")

        library_paths += ["{}/.local/lib/".format(expanduser("~")), "/usr/local/lib/",
                          "/usr/local/lib64/", "/usr/lib/", "/usr/lib64/", 
                          "/usr/lib/x86_64-linux-gnu/"]

        lib_find = False
        lib_set = False
        for item in os.getenv("LD_PRELOAD", "").split(":"):
            if item.endswith('lib{}.so'.format(lib_type)):
                lib_set = True
                break
        if not lib_set:
            for lib_path in library_paths:
                library_file = lib_path + "lib" + lib_type + ".so"
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    if "LD_PRELOAD" in os.environ:
                        os.environ["LD_PRELOAD"] = matches[0] + ":" + os.environ["LD_PRELOAD"]
                    else:
                        os.environ["LD_PRELOAD"] = matches[0]
                    lib_find = True
                    break
        return lib_set or lib_find

    def is_numactl_available(self):
        numactl_available = False
        cmd = ["numactl", "-C", "0", "-m", "0", "ls"]
        r = subprocess.run(cmd, env=os.environ, stdout=subprocess.DEVNULL)
        if r.returncode == 0:
            numactl_available = True
        return numactl_available

    def set_memory_allocator(self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False, benchmark=False):
        '''
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory resue and reduce page fault to improve performance.
        '''
        if enable_tcmalloc and enable_jemalloc:
            logger.error("Unable to enable TCMalloc and JEMalloc at the same time")
            exit(-1)

        if enable_tcmalloc:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if not find_tc:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install -c conda-forge gperftools' to install tcmalloc"
                               .format("TCmalloc", "tcmalloc", expanduser("~")))
            else:
                logger.info("Use TCMalloc memory allocator")

        elif enable_jemalloc:
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if not find_je:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install -c conda-forge jemalloc' to install jemalloc"
                               .format("JeMalloc", "jemalloc", expanduser("~")))
            else:
                logger.info("Use JeMalloc memory allocator")
                if benchmark:
                    self.set_env('MALLOC_CONF', "oversize_threshold:1,background_thread:false,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1")
                else:
                    self.set_env('MALLOC_CONF', "oversize_threshold:1,background_thread:true,metadata_thp:auto")

        elif use_default_allocator:
            pass

        else:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if find_tc:
                logger.info("Use TCMalloc memory allocator")
                return
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if find_je:
                logger.info("Use JeMalloc memory allocator")
                return
            logger.warning("Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                           " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                           "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance"
                           .format(expanduser("~")))

    def logger_env(self, env_name=""):
        if env_name in os.environ:
            logger.info("{}={}".format(env_name, os.environ[env_name]))

    def set_env(self, env_name, env_value=None):
        if not env_value:
            logger.warning("{} is None".format(env_name))
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning("{} in environment variable is {} while the value you set is {}".format(env_name, os.environ[env_name], env_value))
        self.logger_env(env_name)

    # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not. In scenario that use all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores. In this case, KMP_AFFINITY should not be set.
    def set_multi_thread_and_allocator(self, ncore_per_instance, disable_iomp=False, set_kmp_affinity=True, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False, benchmark=False):
        '''
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.
        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benifit.
        '''
        self.set_memory_allocator(enable_tcmalloc, enable_jemalloc, use_default_allocator, benchmark)
        self.set_env("OMP_NUM_THREADS", str(ncore_per_instance))
        if not disable_iomp:
            find_iomp = self.add_lib_preload(lib_type="iomp5")
            if not find_iomp:
                logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib"
                               " or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                               "{}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
                               "you can use 'conda install intel-openmp' to install intel openMP"
                               .format("iomp", "iomp5", expanduser("~")))
            else:
                logger.info("Using Intel OpenMP")
                if set_kmp_affinity:
                    self.set_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
                self.set_env("KMP_BLOCKTIME", "1")
        self.logger_env("LD_PRELOAD")

class MultiInstanceLauncher(Launcher):
    r"""
     Launcher for single instance and multi-instance
     """
    def launch(self, args):
        processes = []
        cores = []
        set_kmp_affinity = True
        enable_taskset = False
        if args.core_list:  # user specify what cores will be used by params
            for core_list_elem in args.core_list.split(","):
                spec_core = core_list_elem.strip()
                if spec_core.isdigit():
                    cores.append(int(spec_core))
                else:
                    core_range = [int(x.strip()) for x in spec_core.split("-")]
                    assert len(core_range) == 2, "Invalid core_list argument format"
                    beg, end = core_range
                    if beg > end:
                        beg, end = end, beg
                    cores.extend(list(range(beg, end + 1)))
            cores = list(set(cores))
            valid_cores = self.cpuinfo.get_all_physical_cores() + self.cpuinfo.get_all_logical_cores()
            for c in cores:
                assert c in valid_cores, "Invalid core ID {} in core_list argument".format(c)

            if args.ncore_per_instance == -1:
                logger.error("please specify the '--ncore_per_instance' if you have pass the --core_list params")
                exit(-1)
            elif args.ninstances > 1 and args.ncore_per_instance * args.ninstances < len(cores):
                logger.warning("only first {} cores will be used, but you specify {} cores in core_list".format(args.ncore_per_instance * args.ninstances, len(cores)))
            else:
                args.ninstances = len(cores) // args.ncore_per_instance

        else:
            if os.environ.get('OMP_NUM_THREADS') is not None:
                if args.node_id != -1:
                    physical_cores_number = len(self.cpuinfo.get_node_physical_cores(args.node_id))
                else:
                    physical_cores_number = len(self.cpuinfo.get_all_physical_cores())
                
                if int(os.environ.get('OMP_NUM_THREADS')) > physical_cores_number:
                    logger.warning("there are {} physical cores but you specify OMP_NUM_THREADS equals to {} ; please set OMP_NUM_THREADS <= physical cores or add argument --use_logical_core to use logical core to meet your setting threads".format(physical_cores_number, int(os.environ.get('OMP_NUM_THREADS')) ))
            if args.use_logical_core:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_logical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_logical_cores()
                    # When using all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores. Thus, KMP_AFFINITY should not be set.
                    set_kmp_affinity = False
            else:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_physical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_physical_cores()
            if not args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
                args.ninstances = 1
                args.ncore_per_instance = len(cores)
            elif args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
                args.throughput_mode = True
            elif args.ncore_per_instance == -1 and args.ninstances != -1:
                if args.ninstances > len(cores):
                    logger.error("there are {} total cores but you specify {} ninstances; please make sure ninstances <= total_cores)".format(len(cores), args.ninstances))
                    exit(-1)
                else:
                    args.ncore_per_instance = len(cores) // args.ninstances
            elif args.ncore_per_instance != -1 and args.ninstances == -1:
                if not args.skip_cross_node_cores:
                    args.ninstances = len(cores) // args.ncore_per_instance
                else:
                    ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
                    num_leftover_cores = ncore_per_node % args.ncore_per_instance
                    if args.ncore_per_instance > ncore_per_node:
                        # too many ncore_per_instance to skip cross-node cores
                        logger.warning("there are {} core(s) per socket, but you specify {} ncore_per_instance and skip_cross_node_cores. Please make sure --ncore_per_instance < core(s) per socket".format(ncore_per_node, args.ncore_per_instance))
                        exit(-1)
                    elif num_leftover_cores == 0:
                        # aren't any cross-node cores
                        logger.info('--skip_cross_node_cores is set, but there are no cross-node cores.')
                        args.ninstances = len(cores) // args.ncore_per_instance
                    else:
                        # skip cross-node cores
                        if args.ninstances != -1:
                            logger.warning('--skip_cross_node_cores is exclusive to --ninstances. --ninstances won\'t take effect even if it is set explicitly.')

                        i = 1
                        leftover_cores = set()
                        while ncore_per_node*i <= len(cores):
                            leftover_cores.update(cores[ncore_per_node*i-num_leftover_cores : ncore_per_node*i])
                            i += 1
                        cores = list(set(cores) - leftover_cores)
                        assert len(cores) % args.ncore_per_instance == 0
                        args.ninstances = len(cores) // args.ncore_per_instance
            else:
                if args.ninstances * args.ncore_per_instance > len(cores):
                    logger.error("Please make sure ninstances * ncore_per_instance <= total_cores")
                    exit(-1)
            if args.latency_mode:
                logger.warning('--latency_mode is exclusive to --ninstances, --ncore_per_instance, --node_id and --use_logical_core. They won\'t take effect even if they are set explicitly.')
                args.ncore_per_instance = 4
                cores = self.cpuinfo.get_all_physical_cores()
                args.ninstances = len(cores) // args.ncore_per_instance

            if args.throughput_mode:
                logger.warning('--throughput_mode is exclusive to --ninstances, --ncore_per_instance, --node_id and --use_logical_core. They won\'t take effect even if they are set explicitly.')
                args.ninstances = self.cpuinfo.node_nums()
                cores = self.cpuinfo.get_all_physical_cores()
                args.ncore_per_instance = len(cores) // args.ninstances

        if args.ninstances > 1 and args.instance_idx != -1:
            logger.info("assigning {} cores for instance {}".format(args.ncore_per_instance, args.instance_idx))

        if not args.disable_numactl:
            numactl_available = self.is_numactl_available()
            if not numactl_available:
                if not args.disable_taskset:
                    logger.warning("Core binding with numactl is not available. Disabling numactl and using taskset instead. This may affect performance in multi-socket system; please use numactl if memory binding is needed.")
                    args.disable_numactl = True
                    enable_taskset = True
                else:
                    logger.warning("Core binding with numactl is not available, and --disable_taskset is set. Please unset --disable_taskset to use taskset insetad of numactl.")
                    exit(-1)

        if not args.disable_taskset:
            enable_taskset = True

        self.set_multi_thread_and_allocator(args.ncore_per_instance,
                                            args.disable_iomp,
                                            set_kmp_affinity,
                                            args.enable_tcmalloc,
                                            args.enable_jemalloc,
                                            args.use_default_allocator,
                                            args.benchmark)
        os.environ["LAUNCH_CMD"] = "#"

        if args.auto_ipex:
            args.program = auto_ipex.apply_monkey_patch(args.program, args.dtype, args.auto_ipex_verbose, args.disable_ipex_graph_mode)

        for i in range(args.ninstances):
            cmd = []
            cur_process_cores = ""
            if not args.disable_numactl or enable_taskset:
                if not args.disable_numactl:
                    cmd = ["numactl"]
                elif enable_taskset:
                    cmd = ["taskset"]

                cores = sorted(cores)
                if args.instance_idx == -1:  # sequentially assign ncores_per_instance to ninstances
                    core_list = cores[i * args.ncore_per_instance: (
                        i + 1) * args.ncore_per_instance]
                else:  # assign ncores_per_instance from instance_idx
                    core_list = cores[args.instance_idx * args.ncore_per_instance: (
                        args.instance_idx + 1) * args.ncore_per_instance]

                core_ranges = []
                for core in core_list:
                    if len(core_ranges) == 0:
                        range_elem = {'start': core, 'end': core}
                        core_ranges.append(range_elem)
                    else:
                        if core - core_ranges[-1]['end'] == 1:
                            core_ranges[-1]['end'] = core
                        else:
                            range_elem = {'start': core, 'end': core}
                            core_ranges.append(range_elem)
                for r in core_ranges:
                    cur_process_cores = cur_process_cores + "{}-{},".format(r['start'], r['end'])
                cur_process_cores = cur_process_cores[:-1]

                if not args.disable_numactl:
                    numa_params = "-C {} ".format(cur_process_cores)
                    numa_params += "-m {}".format(",".join(
                        [str(numa_id) for numa_id in self.cpuinfo.numa_aware_check(core_list)]))
                    cmd.extend(numa_params.split())
                elif enable_taskset:
                    taskset_params = "-c {}".format(cur_process_cores)
                    cmd.extend(taskset_params.split())

            with_python = not args.no_python
            if with_python:
                cmd.append(sys.executable)
                cmd.append("-u")
            if args.module:
                cmd.append("-m")
            cmd.append(args.program)
            log_name = args.log_file_prefix + "_instance_{}_cores_".format(i) + cur_process_cores.replace(',', '_') + ".log"
            log_name = os.path.join(args.log_path, log_name)
            cmd.extend(args.program_args)
            os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
            cmd_s = " ".join(cmd)
            if args.log_path:
                cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_name)
            logger.info(cmd_s)
            if not args.disable_numactl:
                process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
            elif enable_taskset:
                process = subprocess.Popen(cmd, env=os.environ)
            processes.append(process)

            if args.instance_idx != -1: # launches single instance, instance_idx, only
                break

        os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]
        try:
            for process in processes:
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd_s)
        finally:
            if args.auto_ipex:
                # Clean the temp file
                if os.path.exists(args.program) and args.program.endswith("_auto_ipex"):
                    os.remove(args.program)

class DistributedTrainingLauncher(Launcher):
    r"""
     Launcher for distributed traning with MPI launcher
     """
    def get_mpi_pin_domain(self, nproc_per_node, ccl_worker_count, total_cores, logical_core_for_ccl=False):
        '''
        I_MPI_PIN_DOMAIN specify the cores used for every MPI process.
        1)use physical core for oneccl 
        The first ccl_worker_count cores of every rank for ccl communication
        and the other cores will be used to do computation.
        For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
        CCL_WORKER_COUNT=4
        CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
        I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff00000000]
        2)use logical core oneccl 
        The first ccl_worker_count logical cores which is correponding to the 
        first ccl_worker_count physical cores are used as the ccl cores. 
        For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
        CCL_WORKER_COUNT=4
        CCL_WORKER_AFFINITY="56,57,58,59,84,85,86,87"
        I_MPI_PIN_DOMAIN=[0xfffffff,0xfffffff0000000]
        '''
        ppn = nproc_per_node
        cores_per_rank = total_cores // ppn
        pin_domain = "["
        for proc in range(ppn):
            domain_binary = 0
            if logical_core_for_ccl:
                begin = proc * cores_per_rank
            else:
                begin = proc * cores_per_rank + ccl_worker_count
            end = proc * cores_per_rank + cores_per_rank - 1
            for i in range(begin, end + 1):
                domain_binary |= (1 << i)
            pin_domain += hex(domain_binary) + ","
        pin_domain += "]"
        return pin_domain

    def get_ccl_worker_affinity(self, nproc_per_node, 
            ccl_worker_count, total_cores, logical_core_for_ccl=False, physical_to_logical=None):
        '''
        Computation and communication use different cores when using oneCCL
        backend for distributed training.         
        '''
        ppn = nproc_per_node
        cores_per_rank = total_cores // ppn
        affinity = ''
        for proc in range(ppn):
            for ccl_worker in range(ccl_worker_count):
                if logical_core_for_ccl:
                    logical_core = physical_to_logical[proc * cores_per_rank + ccl_worker]
                    affinity += str(logical_core) + ","
                else:
                    affinity += str(proc * cores_per_rank + ccl_worker) + ","
        affinity = affinity[:-1]
        return affinity

    def launch(self, args):
        '''
        Set ENVs and launch MPI process for distributed training.
        '''
        assert not(args.logical_core_for_ccl and args.use_logical_core), "Can't use logical_core_for_ccl and use_logical_core at the same time"
        if args.nnodes > 1 and not os.path.exists(args.hostfile):
            raise ValueError("hostfile is necessary when you use multi-node distributed training,"
                             "Please create hostfile which include the ip list you used for distributed running")
        elif args.nnodes > 1:
            ipv4_addr_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ip_list = []
            with open(args.hostfile) as f:
                for line in f:
                    line = line.strip().strip("\n")
                    # is_valid = re.match(ipv4_addr_pattern, line)
                    # if not is_valid:
                    #     logger.error("{} is not valid IPV4 address".format(line))
                    #     exit(-1)
                    # else:
                    #     ip_list.append(line)
                    ip_list.append(line)
            if len(ip_list) < args.nnodes:
                logger.error("The number of IP {} should greater than nnodes parameters {}".format(len(ip_list), args.nnodes))
                exit(-1)
            master_check = False
            dic = psutil.net_if_addrs()
            for adapter in dic:
                snicList = dic[adapter]
                for snic in snicList:
                    if snic.address == ip_list[0]:
                        master_check = True
            if not master_check:
                logger.error("MASTER_ADDR is incorrect. Please make sure the first line {} in your hostfile is ip address of the current node".format(ip_list[0]))
                exit(-1)

            logger.info("Begin to validate the ip connect")
            args.master_addr = ip_list[0]
            for ip in ip_list[1:]:
                completed_process = subprocess.run("ssh -o PasswordAuthentication=no {} ':'".format(ip), shell=True)
                if completed_process.returncode != 0:
                    logger.error("Passwordless SSH login to {} failed, please make sure you have setup SSH public key right")
                    exit(-1)
                else:
                    logger.info("connection from master node {} to slave node {} is OK".format(args.master_addr, ip))

        total_cores_per_node = self.cpuinfo.physical_core_nums()
        if args.use_logical_core:
            total_cores_per_node = self.cpuinfo.logical_core_nums()

        # set distributed related environmental variables
        self.set_env("MASTER_ADDR", args.master_addr)
        self.set_env("MASTER_PORT", str(args.master_port))
        mpi_pin_domain = self.get_mpi_pin_domain(args.nproc_per_node, 
                                                 args.ccl_worker_count, 
                                                 total_cores_per_node, 
                                                 args.logical_core_for_ccl)
        self.set_env("I_MPI_PIN_DOMAIN", mpi_pin_domain)

        ppn = args.nproc_per_node
        cores_per_rank = total_cores_per_node // ppn
        if args.logical_core_for_ccl:
            omp_num_threads = cores_per_rank
        else:
            omp_num_threads = cores_per_rank - args.ccl_worker_count

        self.set_multi_thread_and_allocator(omp_num_threads,
                                            args.disable_iomp,
                                            True,
                                            args.enable_tcmalloc,
                                            args.enable_jemalloc,
                                            args.use_default_allocator)

        self.set_env("CCL_WORKER_COUNT", str(args.ccl_worker_count))
        ccl_affinity = self.get_ccl_worker_affinity(args.nproc_per_node, 
                                                    args.ccl_worker_count, 
                                                    total_cores_per_node,
                                                    args.logical_core_for_ccl,
                                                    self.cpuinfo.physical_to_logical)
        self.set_env("CCL_WORKER_AFFINITY", ccl_affinity)

        os.environ["LAUNCH_CMD"] = "#"
        cmd = ['mpiexec.hydra']
        mpi_config = "-l -np {} -ppn {} -genv I_MPI_PIN_DOMAIN={} -genv OMP_NUM_THREADS={} ".format(args.nnodes * args.nproc_per_node, args.nproc_per_node, mpi_pin_domain, omp_num_threads)
        mpi_config += args.more_mpi_params
        if args.nnodes > 1:
            mpi_config += " -hostfile {}".format(args.hostfile)
        cmd.extend(mpi_config.split())
        with_python = not args.no_python
        if with_python:
            cmd.append(sys.executable)
            cmd.append("-u")
        if args.module:
            cmd.append("-m")
        cmd.append(args.program)
        cmd.extend(args.program_args)
        log_name = args.log_file_prefix + ".log"
        log_name = os.path.join(args.log_path, log_name)
        cmd_s = " ".join(cmd)
        if args.log_path:
            cmd_s = "{} 2>&1 | tee {}".format(cmd_s, log_name)
        logger.info(cmd_s)
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
        process.wait()
        os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
        os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]

def add_distributed_training_params(parser):

    cpuinfo = CPUinfo()
    node_nums = cpuinfo.node_nums()

    group = parser.add_argument_group("Distributed Training Parameters With oneCCL backend")
    group.add_argument("--nnodes", metavar='\b', type=int, default=1,
                       help="The number of nodes to use for distributed "
                       "training")
    group.add_argument("--nproc_per_node", metavar='\b', type=int, default=node_nums,
                       help="The number of processes to launch on each node")
    # ccl control
    group.add_argument("--ccl_worker_count", metavar='\b', default=4, type=int,
                       help="Core numbers per rank used for ccl communication")
    
    group.add_argument("--logical_core_for_ccl", action='store_true', default=False,
                       help="Use logical core for ccl worker can get better perfomrance in some case.")
    # mpi control
    group.add_argument("--master_addr", metavar='\b', default="127.0.0.1", type=str,
                       help="Master node (rank 0)'s address, should be either "
                            "the IP address or the hostname of node 0, for "
                            "single node multi-proc training, the "
                            "--master_addr can simply be 127.0.0.1")
    group.add_argument("--master_port", metavar='\b', default=29500, type=int,
                       help="Master node (rank 0)'s free port that needs to "
                            "be used for communication during distributed "
                            "training")
    group.add_argument("--hostfile", metavar='\b', default="hostfile", type=str,
                       help="Hostfile is necessary for multi-node multi-proc "
                            "training. hostfile includes the node address list "
                            "node address which should be either the IP address"
                            "or the hostname.")
    group.add_argument("--more_mpi_params", metavar='\b', default="", type=str,
                       help="User can pass more parameters for mpiexec.hydra "
                            "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN")

def add_memory_allocator_params(parser):

    group = parser.add_argument_group("Memory Allocator Parameters")
    # allocator control
    group.add_argument("--enable_tcmalloc", action='store_true', default=False,
                       help="Enable tcmalloc allocator")
    group.add_argument("--enable_jemalloc", action='store_true', default=False,
                       help="Enable jemalloc allocator")
    group.add_argument("--use_default_allocator", action='store_true', default=False,
                       help="Use default memory allocator")

def add_multi_instance_params(parser):

    group = parser.add_argument_group("Multi-instance Parameters")
    # multi-instance control
    group.add_argument("--ncore_per_instance", metavar='\b', default=-1, type=int,
                       help="Cores per instance")
    group.add_argument("--skip_cross_node_cores", action='store_true', default=False,
                       help="If specified --ncore_per_instance, skips cross-node cores.")
    group.add_argument("--ninstances", metavar='\b', default=-1, type=int,
                       help="For multi-instance, you should give the cores number you used for per instance.")
    group.add_argument("--instance_idx", metavar='\b', default="-1", type=int,
                       help="Specify instance index to assign ncores_per_instance for instance_idx; otherwise ncore_per_instance will be assigned sequentially to ninstances. Please refer to https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md")
    group.add_argument("--latency_mode", action='store_true', default=False,
                       help="By detault 4 core per instance and use all physical cores")
    group.add_argument("--throughput_mode", action='store_true', default=False,
                       help="By default one instance per node and use all physical cores")
    group.add_argument("--node_id", metavar='\b', default=-1, type=int,
                       help="node id for multi-instance, by default all nodes will be used")
    group.add_argument("--use_logical_core", action='store_true', default=False,
                       help="Whether only use physical cores")
    group.add_argument("--disable_numactl", action='store_true', default=False,
                       help="Disable numactl")
    group.add_argument("--disable_taskset", action='store_true', default=False,
                       help="Disable taskset")
    group.add_argument("--core_list", metavar='\b', default=None, type=str,
                       help="Specify the core list as 'core_id, core_id, ...' or 'core_id-core_id, ...', otherwise, all the cores will be used.")
    group.add_argument("--benchmark", action='store_true', default=False,
                   help="Enable benchmark config. JeMalloc's MALLOC_CONF has been tuned for low latency. Recommend to use this for benchmarking purpose; for other use cases, this MALLOC_CONF may cause Out-of-Memory crash.")
    group.add_argument("--log_path", metavar='\b', default="", type=str,
                       help="The log file directory. Default path is '', which means disable logging to files.")
    group.add_argument("--log_file_prefix", metavar='\b', default="run", type=str,
                       help="log file prefix")

def add_kmp_iomp_params(parser):

    group = parser.add_argument_group("IOMP Parameters")
    group.add_argument("--disable_iomp", action='store_true', default=False,
                       help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD")

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="This is a script for launching PyTorch training and inference on Intel Xeon CPU "
                                        "with optimal configurations. Now, single instance inference/training, multi-instance "
                                        "inference/training and distributed training with oneCCL backend is enabled. "
                                        "To get the peak performance on Intel Xeon CPU, the script optimizes the configuration "
                                        "of thread and memory management. For thread management, the script configures thread "
                                        "affinity and the preload of Intel OMP library. For memory management, it configures "
                                        "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
                                        "\n################################# Basic usage ############################# \n"
                                        "\n 1. single instance\n"
                                        "\n   >>> ipexrun python_script args \n"
                                        "\n2. multi-instance \n"
                                        "\n    >>> ipexrun --ninstances xxx --ncore_per_instance xx python_script args\n"
                                        "\n3. Single-Node multi-process distributed training\n"
                                        "\n    >>> python  -m intel_extension_for_pytorch.cpu.launch --distributed  python_script args\n"
                                        "\n4. Multi-Node multi-process distributed training: (e.g. two nodes)\n"
                                        "\n   rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*\n"
                                        "\n   >>> ipexrun --distributed --nproc_per_node=2\n"
                                        "\n       --nnodes=2 --hostfile hostfile python_script args\n"
                                        "\n############################################################################# \n",
                                        formatter_class=RawTextHelpFormatter)

    parser.add_argument("--multi_instance", action='store_true', default=False,
                        help="Enable multi-instance, by default one instance per node")

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Enable distributed training.')
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")

    parser.add_argument("--no_python", default=False, action="store_true",
                        help="Do not prepend the --program script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")

    add_memory_allocator_params(parser)
    add_kmp_iomp_params(parser)

    add_distributed_training_params(parser)
    add_multi_instance_params(parser)

    auto_ipex.add_auto_ipex_params(parser)

    # positional
    parser.add_argument("program", type=str,
                        help="The full path to the proram/script to be launched. "
                             "followed by all the arguments for the script")

    # rest from the training program
    parser.add_argument('program_args', nargs=REMAINDER)
    return parser.parse_args()

def main():

    env_before = set(os.environ.keys())
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    args = parse_args()
    if args.log_path:
        path = os.path.dirname(args.log_path if args.log_path.endswith('/') else args.log_path + '/')
        if not os.path.exists(path):
            os.makedirs(path)
        args.log_path = path

        args.log_file_prefix = '{}_{}'.format(args.log_file_prefix, datetime.now().strftime("%Y%m%d%H%M%S"))
        fileHandler = logging.FileHandler("{0}/{1}_instances.log".format(args.log_path, args.log_file_prefix))
        logFormatter = logging.Formatter(format_str)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    if args.distributed and args.multi_instance:
        raise RuntimeError("Either args.distributed or args.multi_instance should be set")

    if args.latency_mode and args.throughput_mode:
        raise RuntimeError("Either args.latency_mode or args.throughput_mode should be set")

    if args.nnodes > 1:
        args.distributed = True

    if not args.no_python and not args.program.endswith(".py"):
        logger.error("For non Python script, you should use '--no_python' parameter.")
        exit()

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
                    logger.warning("{} doesn't exist. Removing it from LD_PRELOAD.".format(item))
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    launcher = None
    if args.distributed:
        launcher = DistributedTrainingLauncher()
    else:
        launcher = MultiInstanceLauncher()

    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug('{0}={1}'.format(x, os.environ[x]))

if __name__ == "__main__":
    main()
