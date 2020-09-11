from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import platform
import subprocess
import os
from argparse import ArgumentParser, REMAINDER

r"""
`intel_pytorch_extension.dist_lauch` is a module that spawns up multiple distributed
training processes on each of the training nodes. For intel_pytorch_extension, oneCCL 
is used as the communication backend and MPI used to lauch multi-proc. To get the better 
performance, you should specify the different cores for oneCCL communication and computation 
process seperately. This tool can automatically set these ENVs(such as I_MPI_PIN_DOMIN) and lauch
multi-proc for you.   

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned.  It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    >>> python -m intel_pytorch_extension.dist_lauch --nproc_per_node=xxx
               YOUR_TRAINING_SCRIPT --arg1 --arg2 --arg3 and all other
               arguments of your training script

2. Multi-Node multi-process distributed training: (e.g. two nodes)


rank 0: *(IP: 192.168.10.10, and has a free port: 295000)*

::

    >>> python -m intel_pytorch_extension.dist_lauch --nproc_per_node=xxx
               --nnodes=2 --hostfile hostfile --master_addr="192.168.10.10"
               --master_port=29500 YOUR_TRAINING_SCRIPT --arg1 --arg2 --arg3 
               and all other arguments of your training script)


3. To look up what optional arguments this module offers:

::

    >>> python -m intel_pytorch_extension.dist_lauch --help

"""

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
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'])
    total_cores = cores_per_socket * sockets
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
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'] )
    total_cores = cores_per_socket * sockets
    cores_per_rank = total_cores // ppn
    affinity = ''
    for proc in range(ppn):
        for ccl_worker in range(args.ccl_worker_count):
            affinity += str(proc * cores_per_rank + ccl_worker)+ "," 
    os.environ["CCL_WORKER_AFFINITY"] = affinity

def get_cpuinfo():
    '''
    Use 'lscpu' command in Linux platform to get the cpu info.
    '''
    origin_info = subprocess.check_output("lscpu", shell=True).strip().decode()
    info_list = origin_info.split("\n")
    info_dict = dict()
    for info in info_list:
        key_value = info.split(":")
        info_dict[key_value[0].strip()] = key_value[1].strip()
    return info_dict

def mpi_dist_lauch(args):
    '''
    Set ENVs and lauch MPI process for distributed training.
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
    
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'] )
    total_cores = cores_per_socket * sockets
    cores_per_rank = total_cores // ppn
    
    if "OMP_NUM_THREADS" not in os.environ:
        opm_num_threads = cores_per_rank - args.ccl_worker_count
    else:
        opm_num_threads = os.environ["OMP_NUM_THREADS"]

    os.environ["CCL_WORKER_COUNT"] = str(args.ccl_worker_count)

    if "CCL_WORKER_AFFINITY" not in os.environ:
        set_ccl_worker_affinity(args)
        
    cmd = ['mpiexec.hydra']
    mpi_config = "-l -np {} -ppn {} -genv I_MPI_PIN_DOMAIN={} -genv OMP_NUM_THREADS={} ".format(args.nnodes*args.nproc_per_node,
                  args.nproc_per_node,  mpi_pin_domain, opm_num_threads)
    mpi_config += args.more_mpi_parms
    if args.nnodes > 1:
        mpi_config += " -hostfile {}".format(args.hostfile)

    cmd.extend(mpi_config.split())
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Torch-ccl distributed training lauch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the lauch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=2,
                        help="The number of processes to lauch on each node")
    parser.add_argument("--ccl_worker_count", default=4, type=int,
                        help="core numbers per rank used for ccl communication")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")

    parser.add_argument("--hostfile", default="hostfile", type=str,
                         help="hostfile is necessary for multi-node multi-proc "
                              "training. hostfile includes the node address list "
                              "node address which should be either the IP address"
                              "or the hostname.")
   
    parser.add_argument("--more_mpi_parms", default="", type=str,
                         help="user can pass more parameters for mpiexec.hydra "
                              "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN")
    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the training script"
                             "program/script to be lauched in parallel, "
                             "followed by all the arguments for the "
                             "training script")
    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    args = parse_args()
    mpi_dist_lauch(args)
 
if __name__ == "__main__":
    main()

