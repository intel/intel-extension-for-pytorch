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

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

1. Run single-instance inference or training on a single node with all CPU sockets.

::

   >>> python -m intel_pytorch_extension.launch script.py args

2. Run single-instance inference or training on a single CPU socket.

::

   >>> python -m intel_pytorch_extension.launch --socket_id 1 script.py args

*** Multi-instance inference *** 

1. Multi-instance 
   By default, one instance per socket. if you want to set the instance numbers and core per instance,  
   --nintances and  --ncore_per_instance should be set. 

   
   >>> python -m intel_pytorch_extension.launch --multi_instance python_script args

   eg: on CLX8280 with 14 instance, 4 cores per instance 
::

   >>> python -m intel_pytorch_extension.launch --multi_instance --nintances 14 --ncore_per_instance 4 python_script args


*** Distributed Training ***

spawns up multiple distributed training processes on each of the training nodes. For intel_pytorch_extension, oneCCL 
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

    >>> python  -m intel_pytorch_extension.launch --distributed  python_script  --arg1 --arg2 --arg3 and all other
                arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*

::

    >>> python -m intel_pytorch_extension.launch --distributed --nproc_per_node=xxx
               --nnodes=2 --hostfile hostfile python_sript --arg1 --arg2 --arg3 
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> python -m intel_pytorch_extension.launch --help

*** Memory allocator  ***

"--enable_tcmalloc" and "--enable_jemalloc" can be used to enable different memory allcator. 

"""

class CPUinfo():
    def __init__(self):

        self.cpuinfo = []
        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
            lscpu_info = subprocess.check_output(args, universal_newlines=True).split("\n")

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]+)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))
            self._get_socket_info()

    def _get_socket_info(self):

        self.socket_physical_cores = [] #socket_id is index
        self.socket_logical_cores = []  #socket_id is index
        self.sockets =  int(max([line[2] for line in self.cpuinfo])) + 1
        for socket_id in range(self.sockets):
            cur_socket_physical_core = []
            cur_socket_logical_core = []
            for line in self.cpuinfo:
                if socket_id == int(line[2]):
                    if line[1] not in cur_socket_physical_core:
                        cur_socket_physical_core.append(line[1])
                    cur_socket_logical_core.append(line[0])
            self.socket_physical_cores.append(cur_socket_physical_core)
            self.socket_logical_cores.append(cur_socket_logical_core)


    def socket_nums(self):
        return self.sockets

    def physical_core_nums(self):
        return len(self.socket_physical_cores) * len(self.socket_physical_cores[0])

    def logical_core_nums(self):
        return len(self.socket_logical_cores) * len(self.socket_logical_cores[0])
    
    def get_socket_physical_cores(self, socket_id):
        if socket_id < 0 or socket_id > self.sockets - 1:
            logger.error("Invalid socket id")
        return self.socket_physical_cores[socket_id]

    def get_socket_logical_cores(self, socket_id):
        if socket_id < 0 or socket_id > self.sockets - 1:
            logger.error("Invalid socket id")
        return self.socket_logical_cores[socket_id]

    def get_all_physical_cores(self):
        return np.array(self.socket_physical_cores).flatten().tolist()
    
    def get_all_logical_cores(self):
        return np.array(self.socket_logical_cores).flatten().tolist()
              

def set_mpi_pin_domain(args):
    '''
    I_MPI_PIN_DOMAIN specify the cores used for every MPI process. 
    The first ccl_worker_count cores of every rank for ccl communication
    and the other cores will be used to do computation.
    For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
    CCL_WORKER_COUNT=4
    CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
    I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff0000000]
    '''
    cpuinfo = CPUinfo()
    ppn = args.nproc_per_node
    total_cores = cpuinfo.physical_core_nums()
    if args.use_logical_core:
        total_cores = cpuinfo.logcal_core_nums()
    cores_per_rank = total_cores // ppn
    pin_domain = "["
    for proc in range(ppn):
        domain_binary = 0
        begin = proc * cores_per_rank + args.ccl_worker_count
        end = proc * cores_per_rank + cores_per_rank -1 
        for i in range(begin, end + 1):
            domain_binary |= (1 << i)
        pin_domain += hex(domain_binary) + ","
    return pin_domain + "]"

def set_ccl_worker_affinity(args):
    '''
    computation and communication use different cores when using oneCCL
    backend for distributed training. we use first ccl_worker_count cores of 
    every rank for ccl communication
    '''
    cpuinfo = CPUinfo()
    ppn = args.nproc_per_node
    total_cores = cpuinfo.physical_core_nums()
    if args.use_logical_core:
        total_cores = cpuinfo.logcal_core_nums()
    cores_per_rank = total_cores // ppn
    affinity = ''
    for proc in range(ppn):
        for ccl_worker in range(args.ccl_worker_count):
            affinity += str(proc * cores_per_rank + ccl_worker)+ "," 
    os.environ["CCL_WORKER_AFFINITY"] = affinity


def add_lib_preload(lib_type=None):
    '''
    Enale TCMalloc/JeMalloc/iomp 
    '''
    library_paths = []
    if "CONDA_PREFIX" in os.environ:
        library_paths.append(os.environ["CONDA_PREFIX"] + "/lib/")
    
    library_paths += ["{}/.local/lib/".format(expanduser("~")), "/usr/local/lib/",
                     "/usr/local/lib64/", "/usr/lib/", "/usr/lib64/"]
    lib_find = False
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
    return lib_find

def set_memory_allocator(args):
    if args.enable_tcmalloc and args.enable_jemalloc:
        logger.error("Unable to enable TCMalloc and JEMalloc at the same time")
        exit(-1)

    if args.enable_tcmalloc: 
        find_tc = add_lib_preload(lib_type="tcmalloc")
        if not find_tc:
            logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or  /.local/lib/"
               " or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
               "~/.local/lib/ so the LD_PRELOAD environment variable will not be set."
               .format("TCmalloc", "tcmalloc", expanduser("~")))
        else:
            logger.info("Use TCMalloc memory allocator")

    elif args.enable_jemalloc:
        find_je = add_lib_preload(lib_type="jemalloc")
        if not find_je:
            logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or  /.local/lib/"
               " or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
               "~/.local/lib/ so the LD_PRELOAD environment variable will not be set."
               .format("JeMalloc", "jemalloc", expanduser("~")))
        else:
            logger.info("Use JeMallocl memory allocator")

    elif args.use_default_allocator:
        pass

    else:
        find_tc = add_lib_preload(lib_type="tcmalloc")
        if find_tc:
            logger.info("Use TCMalloc memory allocator")
            return 
        find_je = add_lib_preload(lib_type="jemalloc")
        if find_je:
            logger.info("Use JeMallocl memory allocator")
            return 
        logger.warning("Both TCMalloc and JeMalloc are not fount in $CONDA_PREFIX/lib or  /.local/lib/"
                       " or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
                       "~/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance"
                       .format(expanduser("~")))
         
def set_multi_thread_and_allcator(args):
    
    set_memory_allocator(args)
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.ncore_per_instance)
    elif "OMP_NUM_THREADS" in os.environ:
        args.ncore_per_instance = int(os.environ["OMP_NUM_THREADS"])
    
    if "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = args.kmp_affinity
    
    if "KMP_BLOCKTIME" not in os.environ:
        os.environ["KMP_BLOCKTIME"] = "1"
    
    if "DNNL_PRIMITIVE_CACHE_CAPACITY" not in os.environ:    
       os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = '1024'

    logger.info("OMP_NUM_THREADS={} ".format(os.environ["OMP_NUM_THREADS"]))
    logger.info("KMP_AFFINITY={}".format(os.environ["KMP_AFFINITY"]))
    logger.info("KMP_BLOCKTIME={}".format(os.environ["KMP_BLOCKTIME"]))
    logger.info("DNNL_PRIMITIVE_CACHE_CAPACITY={}".format(os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"]))
     
    if args.enable_iomp:
        find_iomp = add_lib_preload(lib_type="iomp")
        if not find_iomp:
            logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or  /.local/lib/"
               " or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
               "~/.local/lib/ so the LD_PRELOAD environment variable will not be set."
               .format("iomp", "iomp", expanduser("~")))
        else:
            logger.info("User iomp") 
 
def launch(args):
    '''
    single-instance / multi-instance launcher  
    ''' 
    processes = []
    cores = []
 
    cpuinfo = CPUinfo()
    if args.core_list:#user specify what cores will be used by params
        cores = args.core_list.strip().split(",")
        if args.ncore_per_instance == -1:
            logger.error("please specify the '--ncore_per_instance' if you have pass the --core_list params")
            exit(-1) 
        elif args.ninstances > 1 and args.ncore_per_instance * args.ninstances < len(cores):
            logger.warning("only first {} cores will be used, but you specify {} cores in core_list".format
                  (args.ncore_per_instance * args.ninstances, len(cores)))
        else:
            args.ninstances = len(cores) // args.ncore_per_instance
    else:
        if args.use_logical_core:
            if args.socket_id != -1:
                cores = cpuinfo.get_socket_logical_cores(args.socket_id) 
            else:
                cores = cpuinfo.get_all_logical_cores()            
        else:
            if args.socket_id != -1:
                cores = cpuinfo.get_socket_physical_cores(args.socket_id)
            else:
                cores = cpuinfo.get_all_physical_cores()      
        if not args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
            args.ninstances = 1;
            args.ncore_per_instance = len(cores)
        elif args.multi_instance and args.ninstances == -1 and args.ncore_per_instance == -1:
            args.throughput_performance = True
        elif args.ncore_per_instance == -1 and args.ninstances != -1:
            args.ncore_per_instance = len(cores) // args.ninstances
        elif args.ncore_per_instance != -1 and args.ninstances == -1:
            args.ninstances = len(cores) // args.ncore_per_instance
        else:
            if args.ninstances * args.ncore_per_instance > len(cores):
                logger.error("Please make sure ninstances * ncore_per_instance <= total_cores")
                exit(-1)
        if args.latency_performance:
            if args.ncore_per_instance !=4:
               logger.warning("latency_performance is a specail mode, args.ncore_per_instance can only be set to be 4")
            args.ncore_per_instance = 4
            cores = cpuinfo.get_all_physical_cores()
            args.ninstances = len(cores) // args.ncore_per_instance

        if args.throughput_performance:
            args.ninstances = cpuinfo.socket_nums()
            cores = cpuinfo.get_all_physical_cores()
            args.ncore_per_instance = len(cores) // args.ninstances

    os.environ["LAUNCH_CMD"] = "#"
    set_multi_thread_and_allcator(args)
    for i in range(args.ninstances):
       cmd = []
       cur_process_cores = ""
       if not args.disable_numactl:
           cmd = ["numactl"]
           for core in cores[i * args.ncore_per_instance:(i + 1) * args.ncore_per_instance]:
               cur_process_cores = cur_process_cores + str(core) + ","
           numa_params = "-C {} ".format(cur_process_cores[:-1])
           cmd.extend(numa_params.split())
       with_python = not args.no_python
       if with_python:
           cmd.append(sys.executable)
       if args.module:
           cmd.append("-m")
       cmd.append(args.program)
       cmd.extend(args.program_args)
       os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
       process = subprocess.Popen(cmd, env=os.environ)
       processes.append(process)
    os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd) 
    
def mpi_dist_launch(args):
    '''
    Set ENVs and launch MPI process for distributed training.
    '''
    if args.nnodes > 1 and not os.path.exists(args.hostfile):
        raise ValueError("hostfile is necessary when you use multi-node distributed training,"
                          "Please create hostfile which include the ip list you used for distributed running")
    elif args.nnodes > 1:
        ipv4_addr_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        ip_list = []
        with open(args.hostfile) as f:
             for line in f:
                 line = line.strip().strip("\n")
                 is_valid = re.match(ipv4_addr_pattern, line)
                 if not is_valid:
                     logger.error("{} is not valid IPV4 address".format(line))
                     exit(-1)
                 else:
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
           logger.error("MASTER_ADDR is not right. Please make sure the first ip {} in your hostfile is the current node".format(ip_list[0]))
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

    set_memory_allocator(args)
    # set distributed related environmental variables
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    if "I_MPI_PIN_DOMAIN" not in os.environ:
         mpi_pin_domain = set_mpi_pin_domain(args)
    else:
         mpi_pin_domain = os.environ["I_MPI_PIN_DOMAIN"]
    
    cpuinfo = CPUinfo()
    ppn = args.nproc_per_node 
    total_cores = len(cpuinfo.get_all_physical_cores())
    cores_per_rank = total_cores // ppn
    
    if "OMP_NUM_THREADS" not in os.environ:
        opm_num_threads = cores_per_rank - args.ccl_worker_count
    else:
        opm_num_threads = os.environ["OMP_NUM_THREADS"]

    os.environ["CCL_WORKER_COUNT"] = str(args.ccl_worker_count)

    if "CCL_WORKER_AFFINITY" not in os.environ:
        set_ccl_worker_affinity(args)

    if "CCL_ATL_TRANSPORT" not in os.environ:
        os.environ["CCL_ATL_TRANSPORT"] = "ofi"
    
    if args.enable_iomp:
        find_iomp = add_lib_preload(lib_type="iomp")
        if not find_iomp:
            logger.warning("Unable to find the {} library file lib{}.so in $CONDA_PREFIX/lib or  /.local/lib/"
               " or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or "
               "~/.local/lib/ so the LD_PRELOAD environment variable will not be set."
               .format("iomp", "iomp", expanduser("~")))
        else:
             logger.info("Enale iomp by set LD_PRELOAD")

    logger.info("MASTER_ADDR={}".format(args.master_addr))
    logger.info("MASTER_PORT={}".format(args.master_port))
    logger.info("I_MPI_PIN_DOMAIN={}".format(mpi_pin_domain))
    logger.info("OMP_NUM_THREADS={} ".format(opm_num_threads))
    logger.info("CCL_WORKER_COUNT={}".format(args.ccl_worker_count))
    logger.info("CCL_WORKER_AFFINITY={}".format(os.environ["CCL_WORKER_AFFINITY"]))

    os.environ["LAUNCH_CMD"] = "#"
    cmd = ['mpiexec.hydra']
    mpi_config = "-l -np {} -ppn {} -genv I_MPI_PIN_DOMAIN={} -genv OMP_NUM_THREADS={} ".format(args.nnodes*args.nproc_per_node,
                  args.nproc_per_node,  mpi_pin_domain, opm_num_threads)
    mpi_config += args.more_mpi_parms
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
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()
    os.environ["LAUNCH_CMD"] += " ".join(cmd) + ",#"
    os.environ["LAUNCH_CMD"] = os.environ["LAUNCH_CMD"][:-2]

def add_distributed_training_params(parser):
    
    cpuinfo = CPUinfo()
    socket_nums = cpuinfo.socket_nums()

    group = parser.add_argument_group("Distributed Training Parameters With oneCCL backend")
    group.add_argument("--nnodes", metavar='\b', type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    group.add_argument("--nproc_per_node", metavar='\b', type=int, default=socket_nums,
                        help="The number of processes to launch on each node")
    #ccl control 
    group.add_argument("--ccl_worker_count", metavar='\b', default=4, type=int,
                        help="Core numbers per rank used for ccl communication")
    #mpi control
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
    group.add_argument("--more_mpi_parms", metavar='\b', default="", type=str,
                        help="User can pass more parameters for mpiexec.hydra "
                              "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN")

def add_memory_allocator_params(parser):

    group = parser.add_argument_group("Memory Allocator Parameters") 
        #allocator control
    group.add_argument("--enable_tcmalloc", action='store_true', default=False,
                        help="Enable tcmalloc allocator")
    group.add_argument("--enable_jemalloc", action='store_true', default=False,
                        help="Enable jemalloc allocator")
    group.add_argument("--use_default_allocator",  action='store_true', default=False,
                        help="Use default memory allocator")
        
def add_multi_instance_params(parser):
    
    group = parser.add_argument_group("Multi-instance Parameters")
     #multi-instance control
    group.add_argument("--ncore_per_instance", metavar='\b', default=-1, type=int, 
                         help="Cores per instance")
    group.add_argument("--ninstances", metavar='\b', default=-1, type=int,
                         help="For multi-instance, you should give the cores number you used for per insantance.")
    group.add_argument("--latency_performance", action='store_true', default=False,
                         help="By detault 4 core per instance and use all physical cores")
    group.add_argument("--throughput_performance", action='store_true', default=False,
                         help="By default one instance per socket and use all physical cores")
    group.add_argument("--socket_id", metavar='\b', default=-1, type=int,
                         help="Socket id for multi-instance, by default all sockets will be used")
    group.add_argument("--use_logical_core", action='store_true', default=False,
                         help="Whether only use physical cores")
    group.add_argument("--disable_numactl",  action='store_true', default=False,
                         help="Disable numactl")
    group.add_argument("--core_list", metavar='\b', default=None, type=str,
                         help="Specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used.")
 
def add_kmp_iomp_params(parser): 

    group = parser.add_argument_group("KMP/IOMP Affinity Parameters") 
    group.add_argument("--kmp_affinity", metavar='\b', default="granularity=fine,compact,1,0", type=str,
                        help="KMP_AFFINITY setup, environment variable has higher priority than this args."
                             "defualt value is : granularity=fine,compact,1,0")
    group.add_argument("--enable_iomp", action='store_true', default=False,
                        help="Enable iomp and libiomp.so will be add to LD_PRELOAD") 
   

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
                                         "\n   >>> python -m intel_pytorch_extension.launch python_script args \n"
                                        "\n2. multi-instance \n"
                                        "\n    >>> python -m intel_pytorch_extension.launch --multi_instance python_script args\n"
                                        "\n3. Single-Node multi-process distributed training\n"
                                        "\n    >>> python  -m intel_pytorch_extension.launch --distributed  python_script args\n"
                                        "\n4. Multi-Node multi-process distributed training: (e.g. two nodes)\n"
                                        "\n   rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*\n"
                                        "\n   >>> python -m intel_pytorch_extension.launch --distributed --nproc_per_node=2\n"
                                        "\n       --nnodes=2 --hostfile hostfile python_script args\n",
                                        formatter_class=RawTextHelpFormatter)
    
    parser.add_argument("--multi_instance", action='store_true', default=False,
                        help="Enable multi-instance, by default one instance per socket")  

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

    if args.distributed and args.multi_instance:
        raise RuntimeError("Either args.distributed or args.multi_instance should be set")
    
    if args.latency_performance and args.throughput_performance:
        raise RuntimeError("Either args.latency_performance or args.throughput_performance  should be set")

    if args.nnodes > 1:
        args.distributed = True

    if args.distributed:
        mpi_dist_launch(args)
    else:
        launch(args)

    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug(f'{x}={os.environ[x]}')
 
if __name__ == "__main__":
    main()

