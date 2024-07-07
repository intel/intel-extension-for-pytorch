import re
import sys
import subprocess
import os
import psutil
from .launcher_base import Launcher
from ...utils._logger import WarningType


class DistributedTrainingLauncher(Launcher):
    """
    Launcher for distributed traning with MPI launcher
    """

    def add_params(self, parser):
        group = parser.add_argument_group(
            "Distributed Training Arguments With oneCCL backend"
        )
        group.add_argument(
            "--nnodes",
            type=int,
            default=0,
            help="Number of machines/devices to use for distributed training",
        )
        group.add_argument(
            "--nprocs-per-node",
            "--nprocs_per_node",
            type=int,
            default=0,
            help="Number of processes run on each machine/device",
        )
        # ccl control
        group.add_argument(
            "--ccl-worker-count",
            "--ccl_worker_count",
            default=4,
            type=int,
            help="Number of cores per rank for ccl communication",
        )
        group.add_argument(
            "--logical-cores-for-ccl",
            "--logical_cores_for_ccl",
            action="store_true",
            default=False,
            help="Use logical cores for the ccl worker.",
        )
        # mpi control
        group.add_argument(
            "--master-addr",
            "--master_addr",
            default="127.0.0.1",
            type=str,
            help="Address of master node (rank 0), should be either "
            "IP address or hostname of node 0. For "
            "single node multi-proc training, the "
            "--master-addr can simply be 127.0.0.1",
        )
        group.add_argument(
            "--master-port",
            "--master_port",
            default=29500,
            type=int,
            help="Port on master node (rank 0) for communication during distributed training",
        )
        group.add_argument(
            "--hostfile",
            default="hostfile",
            type=str,
            help="Set the hostfile for multi-node multi-proc "
            "training. The hostfile includes a node address list "
            "containing either IP addresses or hostnames of computation nodes.",
        )
        group.add_argument(
            "--extra-mpi-params",
            "--extra_mpi_params",
            default="",
            type=str,
            help="Extra parameters for mpiexec.hydra "
            "except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN",
        )

    def get_pin_domain_affinity(
        self, cpu_pools, ccl_worker_count, logical_cores_for_ccl=False
    ):
        """
        I_MPI_PIN_DOMAIN specify the cores used for every MPI process.
        1) use physical core for oneccl
           The first ccl_worker_count cores of every rank for ccl communication
           and the other cores will be used to do computation.
           For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
           CCL_WORKER_COUNT=4
           CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31"
           I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff00000000]
        2) use logical core oneccl
           The first ccl_worker_count logical cores which is correponding to the
           first ccl_worker_count physical cores are used as the ccl cores.
           For example: on CascadeLake 8280 CPU, 2 ranks on one node. ccl_worker_count=4
           CCL_WORKER_COUNT=4
           CCL_WORKER_AFFINITY="56,57,58,59,84,85,86,87"
           I_MPI_PIN_DOMAIN=[0xfffffff,0xfffffff0000000]
        """
        domain_binaries = []
        affinity = []
        for pool in cpu_pools:
            if (
                logical_cores_for_ccl
                and len([c for c in pool if not c.is_physical_core]) < ccl_worker_count
            ):
                self.verbose(
                    "warning",
                    "Argument --logical-cores-for-ccl is set but no enough logical cores are available. Disable this argument."
                    + "please see https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/launch_script.html#launch-script-usage-guide"  # noqa: B950
                    + "for usage guide",
                    warning_type=WarningType.WrongArgument,
                )
                logical_cores_for_ccl = False
                break
        for pool in cpu_pools:
            domain_binary = 0
            cores = []
            if logical_cores_for_ccl:
                affinity.extend(
                    [str(c.cpu) for c in pool if not c.is_physical_core][
                        :ccl_worker_count
                    ]
                )
                cores = [str(c.cpu) for c in pool if c.is_physical_core]
            else:
                physical_cores = [str(c.cpu) for c in pool if c.is_physical_core]
                assert ccl_worker_count < len(
                    physical_cores
                ), f"ccl_worker_count ({ccl_worker_count}) cannot exceed number of available cores ({len(physical_cores)})."
                affinity.extend(physical_cores[:ccl_worker_count])
                cores = physical_cores[ccl_worker_count:]
            for c in cores:
                domain_binary |= 1 << int(c)
            domain_binaries.append(hex(domain_binary))
        return {
            "pin_domain": f'[{",".join(domain_binaries)}]',
            "affinity": ",".join(affinity),
        }

    def launch(self, args):
        """
        Set ENVs and launch MPI process for distributed training.
        """
        assert not (
            args.logical_cores_for_ccl and args.use_logical_cores
        ), "Can't use --logical-cores-for-ccl and --use-logical-cores at the same time."
        if args.nnodes > 1:
            assert os.path.exists(args.hostfile), (
                "A hostfile is required when you perform multi-node distributed training. "
                + "Please create the hostfile which includes ip addresses of nodes that you will "
                + "use for the distributed computation workload."
            )
            ipv4_addr_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ip_list = []
            with open(args.hostfile) as f:
                for line in f:
                    line = line.strip().strip("\n")
                    # is_valid = re.match(ipv4_addr_pattern, line)
                    # if not is_valid:
                    #     logger.error('{} is not valid IPV4 address'.format(line))
                    #     exit(-1)
                    # else:
                    #     ip_list.append(line)
                    ip_list.append(line)
            assert (
                len(ip_list) >= args.nnodes
            ), f"Number of IP addresses in the hostfile ({len(ip_list)}) should be >= the nnodes argument ({args.nnodes})."
            master_check = False
            dic = psutil.net_if_addrs()
            for adapter in dic:
                snicList = dic[adapter]
                for snic in snicList:
                    if snic.address == ip_list[0]:
                        master_check = True
            assert master_check, (
                f"MASTER_ADDR is incorrect. Please make sure the first line ({ip_list[0]}) of the hostfile "
                + "is the ip address of the current node."
            )

            self.verbose("info", "Begin to validate SSH connections")
            args.master_addr = ip_list[0]
            for ip in ip_list[1:]:
                completed_process = subprocess.run(
                    f'ssh -o PasswordAuthentication=no {ip} ":"', shell=True
                )
                assert (
                    completed_process.returncode == 0
                ), f"Passwordless SSH login to {ip} failed, please make sure you have a SSH public key setup correctly."
                self.verbose(
                    "info",
                    f"Connection from the master node {args.master_addr} to the slave node {ip} succeeded.",
                )

        nodes_list = self.parse_list_argument(args.nodes_list)
        if args.nprocs_per_node == 0:
            args.nprocs_per_node = (
                len(set([c.node for c in self.cpuinfo.pool_all]))
                if len(nodes_list) == 0
                else len(nodes_list)
            )
        ncores_per_instance = args.ncores_per_instance
        if ncores_per_instance > 0:
            if (
                not args.logical_cores_for_ccl
                or len([c for c in self.cpuinfo.pool_all if not c.is_physical_core])
                < args.nprocs_per_node * args.ccl_worker_count
            ):
                ncores_per_instance += args.ccl_worker_count
            ncores_per_instance = len(
                [c for c in self.cpuinfo.pool_all if c.core < ncores_per_instance]
            )
        self.cpuinfo.gen_pools_ondemand(
            ninstances=args.nprocs_per_node,
            ncores_per_instance=ncores_per_instance,
            use_logical_cores=True,
            use_e_cores=args.use_e_cores,
            nodes_list=nodes_list,
            strategy=args.strategy,
            bind_numa_node=args.bind_numa_node,
        )

        self.set_memory_allocator(args.memory_allocator, False, ["jemalloc"])
        self.set_omp_runtime(args.omp_runtime, True)
        ninstances = len(self.cpuinfo.pools_ondemand)
        omp_num_threads = []
        for i in range(ninstances):
            omp_num_threads.append(
                len([c for c in self.cpuinfo.pools_ondemand[i] if c.is_physical_core])
            )
        omp_num_threads_value = min(omp_num_threads)
        if not args.logical_cores_for_ccl:
            omp_num_threads_value -= args.ccl_worker_count
        self.add_env("OMP_NUM_THREADS", str(omp_num_threads_value))

        # set distributed related environmental variables
        self.add_env("MASTER_ADDR", args.master_addr)
        self.add_env("MASTER_PORT", str(args.master_port))
        pin_domain_affinity = self.get_pin_domain_affinity(
            self.cpuinfo.pools_ondemand,
            args.ccl_worker_count,
            args.logical_cores_for_ccl,
        )
        self.add_env("I_MPI_PIN_DOMAIN", pin_domain_affinity["pin_domain"])
        self.add_env("CCL_WORKER_COUNT", str(args.ccl_worker_count))
        self.add_env("CCL_WORKER_AFFINITY", pin_domain_affinity["affinity"])

        ld_preload_marker = "LD_PRELOAD_UNSET"
        ld_preload_bk = (
            os.environ["LD_PRELOAD"]
            if "LD_PRELOAD" in os.environ
            else ld_preload_marker
        )
        if len(self.ld_preload) > 0:
            os.environ["LD_PRELOAD"] = ":".join(self.ld_preload)
            self.verbose("info", f'LD_PRELOAD={os.environ["LD_PRELOAD"]}')
        else:
            if "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
        for k, v in self.environ_set.items():
            self.verbose("info", f"env: {k}={v}")

        cmd = ["mpiexec.hydra"]
        genvs = [f"-genv {k}={v}" for k, v in self.environ_set.items()]
        mpi_config = f"-l -np {args.nnodes * args.nprocs_per_node} -ppn {args.nprocs_per_node} {' '.join(genvs)} "
        mpi_config += args.extra_mpi_params
        if args.nnodes > 1:
            mpi_config += f" -hostfile {args.hostfile}"
        cmd.extend(mpi_config.split())
        with_python = not args.no_python
        if with_python:
            cmd.append(sys.executable)
            cmd.append("-u")
        if args.module:
            cmd.append("-m")
        cmd.append(args.program)
        cmd.extend(args.program_args)
        log_name = f"{args.log_file_prefix}.log"
        log_name = os.path.join(args.log_dir, log_name)
        cmd_s = " ".join(cmd)
        if args.log_dir:
            cmd_s = f"{cmd_s} 2>&1 | tee {log_name}"
        self.verbose("info", f"cmd: {cmd_s}")
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
        process.wait()
        if ld_preload_bk == ld_preload_marker:
            if "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
        else:
            os.environ["LD_PRELOAD"] = ld_preload_bk
        if args.log_dir:
            log_fns = []
            for i in range(args.nnodes * args.nprocs_per_node):
                log_name_rank = f"{args.log_file_prefix}_rank_{i}.log"
                log_name_rank = os.path.join(args.log_dir, log_name_rank)
                fn = open(log_name_rank, "w")
                log_fns.append(fn)
            with open(log_name) as fp:
                for line in fp:
                    m = re.match(r"\[(\d+)\] (.*)", line.strip())
                    if m:
                        log_fns[int(m.group(1))].write(f"{m.group(2)}\n")
                    else:
                        self.verbose(
                            "warning",
                            f'Failed to detect rank id from log file {log_name} at line "{line.strip()}".',
                            warning_type=WarningType.NotSupported,
                        )
            for fn in log_fns:
                fn.close()


if __name__ == "__main__":
    pass
