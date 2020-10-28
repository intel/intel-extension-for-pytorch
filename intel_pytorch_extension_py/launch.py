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

r"""
This is a module is lanchuer script which can help you launch your program for training and inference.
and automaitically to setup  strategy of thread affinity for multi-thread.The function as following:

1. multi-thread setup

   1) KMP_AFFINITY 
   2) OMP_NUM_THREADS

2. memory allocator enable

   TCMalloc/JeMalloc

3. MPI multi-process launch for oneCCL backend distributed training and the following environment 
   variable will be setup

   1) CCL_WORKER_AFFINITY 
   2) I_MPI_PIN_DOMAIN  
   3) CCL_WORKER_COUNT

4. multi-instance launch and core bind with numactl 
 
**How to use this module:**

Distributed Training 

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

    >>> python  -m intel_pytorch_extension.launch --distributed  --nproc_per_node=xxx
               YOUR_TRAINING_SCRIPT --arg1 --arg2 --arg3 and all other
               arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*

::

    >>> python -m intel_pytorch_extension.launch --distributed --nproc_per_node=xxx
               --nnodes=2 --hostfile hostfile --master_addr="192.168.10.10"
               --master_port=29500 YOUR_TRAINING_SCRIPT --arg1 --arg2 --arg3 
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> python -m intel_pytorch_extension.launch --help

Infrence/multi-instance

for inference and no-distributed training. in default, only one instance will be run for your program. 
For multi-instance, you can specify the instance number and cores per instance by "--ninstances" and "--ncore_per_instance" 
Beside, the numctl is enabled to setup the cores for every instance, and you can disable it by "--disable_numactl". 
For numactl, you can specify the core_list for all the instance and this utlity will distribute the core resource automatically.
otherwise, all the physical cores will be used. 
    eg: on CLX8280 with 2 instance 
    >>> python -m intel_pytorch_extension.launch --nintances 2 --ncore_per_instance 28 xxx.py xxxxxxx
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
           print("Error: invalid socket id")
        return self.socket_physical_cores[socket_id]

    def get_socket_logical_cores(self, socket_id):
        if socket_id < 0 or socket_id > self.sockets - 1:
           print("Error: invalid socket id")
        return self.socket_logical_cores[socket_id]

    def get_all_physical_cores(self):
        return np.array(self.socket_physical_cores).flatten().tolist()
    
    def get_all_logical_cores(self):
        return np.array(self.socket_logical_cores).flatten().tolist()
              

def set_mpi_pin_domain(args):
    '''
    I_MPI_PIN_DOMAIN specify the cores used for every MPI process. 
    The firt ccl_worker_count cores of every rank for ccl communication
    and the other cores will be used to do commputation.
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
    backend for distributed training. we use firt ccl_worker_count cores of 
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


def enable_allocator(args):
    '''
    Enale TCMalloc/JeMalloc according to allocator_type
    xxx.so file can be only in /usr/lib  /usr/lib64 or  ~/.local/lib/ directory 
    '''
    if args.enable_tcmalloc and args.enable_jemalloc:
        print("Error: Unable to enable tcmalloc and jemalloc at the same time")
        exit(-1)
    library_paths = ["/usr/lib/", "/usr/lib64/", "{}/.local/lib/".format(expanduser("~"))]
    allocator_type = None
    lib_find = False
    if args.enable_tcmalloc:
       allocator_type = "tcmalloc"
    elif args.enable_jemalloc:
       allocator_type = "jemalloc"
    for lib_path in library_paths:
        library_file = lib_path + "lib" + allocator_type + ".so"
        matches = glob.glob(library_file)
        if len(matches) > 0:
            os.environ["LD_PRELOAD"] = matches[0]
            lib_find = True
            break
    if not lib_find:
        # Unable to find the TCMalloc library file
        print("Warning: Unable to find the {} library file lib{}.so) in /usr/lib or /usr/lib64"
               "or ~/.local/lib/ so the LD_PRELOAD environment variable will not be set.".format(allocator_type, allocator_type))
    
def set_multi_thread_and_allcator(args):

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.ncore_per_instance)
    elif "OMP_NUM_THREADS" in os.environ:
        args.ncore_per_instance = int(os.environ["OMP_NUM_THREADS"])
        ###to add warining if args.ncore_per_instance != os.environ["OMP_NUM_THREADS"]
    
    if "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = args.kmp_affinity
    
    os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = '1024'

    if args.enable_tcmalloc or args.enable_jemalloc: 
        enable_allocator(args)

 
def launch(args):
    processes = []
    cores = []
 
    cpuinfo = CPUinfo()
    if args.core_list:#user specify what cores will be used by params
        cores = args.core_list.strip().split(",")
        if args.ncore_per_instance == -1:
            print("please specify the '--ncore_per_instance' if you have pass the --core_list params")
            exit(-1) 
        elif args.ninstances > 1 and args.ncore_per_instance * args.ninstances < len(cores):
            print("only first {} cores will be used, but you specify {} cores in core_list".format
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
        if args.ncore_per_instance == -1:
            args.ncore_per_instance = len(cores) // args.ninstances

    set_multi_thread_and_allcator(args)

    for i in range(args.ninstances):
       cmd = []
       cur_process_cores = ""
       if args.disable_numactl: 
           cmd.append(args.program)
           cmd.extend(args.program_args)
       else:
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
       process = subprocess.Popen(cmd, env=os.environ)
       processes.append(process)

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
                          "Please create hostfile which include the ip list you used for distributed runing")

    # set distributed related environmental variables
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    
    if "I_MPI_PIN_DOMAIN" not in os.environ:
         mpi_pin_domain = set_mpi_pin_domain(args)
    else:
         mpi_pin_domain = os.environ["I_MPI_PIN_DOMAIN"]
    
    cpuinfo = CPUinfo()
    ppn = args.nproc_per_node 
    sockets = cpuinfo.socket_nums()
    total_cores = len(cpuinfo.get_all_physical_cores())
    cores_per_rank = total_cores // ppn
    
    if "OMP_NUM_THREADS" not in os.environ:
        opm_num_threads = cores_per_rank - args.ccl_worker_count
    else:
        opm_num_threads = os.environ["OMP_NUM_THREADS"]

    os.environ["CCL_WORKER_COUNT"] = str(args.ccl_worker_count)

    if "CCL_WORKER_AFFINITY" not in os.environ:
        set_ccl_worker_affinity(args)

    if args.enable_tcmalloc or args.enable_jemalloc:
        enable_allocator(args)
        
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
    if args.module:
        cmd.append("-m")
    cmd.append(args.program)
    cmd.extend(args.program_args)
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()

def add_distributed_training_params(parser):
    group = parser.add_argument_group("Distributed Training Parameters With oneCCL backend")
    group.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    group.add_argument("--nproc_per_node", type=int, default=2,
                        help="The number of processes to launch on each node")
    #ccl control 
    group.add_argument("--ccl_worker_count", default=4, type=int,
                        help="core numbers per rank used for ccl communication")
    #mpi control
    group.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    group.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    group.add_argument("--hostfile", default="hostfile", type=str,
                        help="hostfile is necessary for multi-node multi-proc "
                              "training. hostfile includes the node address list "
                              "node address which should be either the IP address"
                              "or the hostname.")
    group.add_argument("--more_mpi_parms", default="", type=str,
                        help="user can pass more parameters for mpiexec.hydra "
                              "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN")

def add_memory_allocator_params(parser):
    group = parser.add_argument_group("Memory Allocator Parameters") 
        #allocator control
    group.add_argument("--enable_tcmalloc", action='store_true', default=False,
                        help="Enable tcmalloc allocator")
    group.add_argument("--enable_jemalloc", action='store_true', default=False,
                        help="Enable jemalloc allocator")
        
def add_multi_instance_params(parser):
    group = parser.add_argument_group("Multi-instance Parameters")
     #multi-instance control  
    group.add_argument("--ncore_per_instance", default=-1, type=int, help="cores per instance")
    group.add_argument("--ninstances", default=1, type=int,
                         help="for multi-instance, you should give the cores number you used for "
                              "per-insantance.")
    group.add_argument("--socket_id", default=-1, type=int,
                         help="socket_id for multi-instance, by default all sockets will be used")
    group.add_argument("--use_logical_core", action='store_true', default=False,
                        help="whether only use physical cores")
    group.add_argument("--disable_numactl", action='store_true', default=False,
                        help="Disable numactl")
    group.add_argument("--core_list", default=None, type=str,
                        help="specify the core list as 'core_id, core_id, ....', otherwise, all the cores will be used.")
 
def add_kmp_params(parser): 
    group = parser.add_argument_group("KMP Affinity Parameters") 
    group.add_argument("--kmp_affinity", default="granularity=fine,compact,1,0", type=str,
                        help="KMP_AFFINITY setup, environment variable has higher priority than this args."
                             "defualt value is : granularity=fine,compact,1,0")
    
   

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Torch-ccl distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")
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
    add_kmp_params(parser)
     
    add_distributed_training_params(parser)
    add_multi_instance_params(parser)
    # positional
    parser.add_argument("program", type=str,
                        help="The full path to the training script"
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('program_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    args = parse_args()
    if args.distributed:
        mpi_dist_launch(args)
    else:
        launch(args)
 
if __name__ == "__main__":
    main()

