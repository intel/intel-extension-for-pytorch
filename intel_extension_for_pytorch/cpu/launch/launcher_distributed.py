import sys
import subprocess
import os
import psutil
from .launcher_base import Launcher

class DistributedTrainingLauncher(Launcher):
    '''
    Launcher for distributed traning with MPI launcher
    '''
    def add_params(self, parser):
        group = parser.add_argument_group('Distributed Training Arguments With oneCCL backend')
        group.add_argument(
            '--nnodes',
            type=int,
            default=0,
            help='Number of machines/devices to use for distributed training',
        )
        group.add_argument(
            '--nprocs-per-node',
            '--nprocs_per_node',
            type=int,
            default=len(set([c.node for c in self.cpuinfo.pool_all])),
            help='Number of processes run on each machine/device',
        )
        # ccl control
        group.add_argument(
            '--ccl-worker-count',
            '--ccl_worker_count',
            default=4,
            type=int,
            help='Number of cores per rank for ccl communication',
        )
        group.add_argument(
            '--logical-cores-for-ccl',
            '--logical_cores_for_ccl',
            action='store_true',
            default=False,
            help='Use logical cores for the ccl worker.',
        )
        # mpi control
        group.add_argument(
            '--master-addr',
            '--master_addr',
            default='127.0.0.1',
            type=str,
            help='Address of master node (rank 0), should be either '
            'IP address or hostname of node 0. For '
            'single node multi-proc training, the '
            '--master-addr can simply be 127.0.0.1',
        )
        group.add_argument(
            '--master-port',
            '--master_port',
            default=29500,
            type=int,
            help='Port on master node (rank 0) for communication during distributed training',
        )
        group.add_argument(
            '--hostfile',
            default='hostfile',
            type=str,
            help='Set the hostfile for multi-node multi-proc '
            'training. The hostfile includes a node address list '
            'containing either IP addresses or hostnames of computation nodes.',
        )
        group.add_argument(
            '--extra-mpi-params',
            '--extra_mpi_params',
            default='',
            type=str,
            help='Extra parameters for mpiexec.hydra '
            'except for -np -ppn -hostfile and -genv I_MPI_PIN_DOMAIN',
        )

    def get_pin_domain_affinity(self, cpu_pools, ccl_worker_count, logical_cores_for_ccl=False):
        '''
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
        '''
        domain_binaries = []
        affinity = []
        for pool in cpu_pools:
            domain_binary = 0
            cores = []
            if logical_cores_for_ccl:
                cores = [c for c in pool if c.is_physical_core]
                affinity.extend([str(c.cpu) for c in pool if not c.is_physical_core][:ccl_worker_count])
            else:
                cores = [c for c in pool if c.is_physical_core][ccl_worker_count:]
                affinity.extend([str(c.cpu) for c in pool if c.is_physical_core][:ccl_worker_count])
            for c in cores:
                domain_binary |= (1 << c.cpu)
            domain_binaries.append(hex(domain_binary))
        return {'pin_domain': f'[{",".join(domain_binaries)}]', 'affinity': ','.join(affinity)}

    def launch(self, args):
        '''
        Set ENVs and launch MPI process for distributed training.
        '''
        assert not(args.logical_cores_for_ccl and args.use_logical_cores), 'Can\'t use --logical-cores-for-ccl and --use-logical-cores at the same time.'
        if args.nnodes > 1:
            assert os.path.exists(args.hostfile), 'A hostfile is required when you perform multi-node distributed training. Please create the hostfile which includes ip addresses of nodes that you will use for the distributed computation workload.'
            ipv4_addr_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            ip_list = []
            with open(args.hostfile) as f:
                for line in f:
                    line = line.strip().strip('\n')
                    # is_valid = re.match(ipv4_addr_pattern, line)
                    # if not is_valid:
                    #     logger.error('{} is not valid IPV4 address'.format(line))
                    #     exit(-1)
                    # else:
                    #     ip_list.append(line)
                    ip_list.append(line)
            assert len(ip_list) >= args.nnodes, f'Number of IP addresses in the hostfile ({len(ip_list)}) should be >= the nnodes argument ({args.nnodes}).'
            master_check = False
            dic = psutil.net_if_addrs()
            for adapter in dic:
                snicList = dic[adapter]
                for snic in snicList:
                    if snic.address == ip_list[0]:
                        master_check = True
            assert master_check, 'MASTER_ADDR is incorrect. Please make sure the first line ({ip_list[0]}) of the hostfile is the ip address of the current node.'

            self.verbose('info', 'Begin to validate SSH connections')
            args.master_addr = ip_list[0]
            for ip in ip_list[1:]:
                completed_process = subprocess.run(f'ssh -o PasswordAuthentication=no {ip} ":"', shell=True)
                assert completed_process.returncode == 0, f'Passwordless SSH login to {ip} failed, please make sure you have a SSH public key setup correctly.'
                self.verbose('info', f'Connection from the master node {args.master_addr} to the slave node {ip} succeeded.')

        self.cpuinfo.gen_pools_ondemand(ninstances=args.nprocs_per_node, use_logical_cores=True)

        # set distributed related environmental variables
        self.set_env('MASTER_ADDR', args.master_addr)
        self.set_env('MASTER_PORT', str(args.master_port))
        pin_domain_affinity = self.get_pin_domain_affinity(
                self.cpuinfo.pools_ondemand,
                args.ccl_worker_count,
                args.logical_cores_for_ccl,
                )
        self.set_env('I_MPI_PIN_DOMAIN', pin_domain_affinity['pin_domain'])
        self.set_env('CCL_WORKER_COUNT', str(args.ccl_worker_count))
        self.set_env('CCL_WORKER_AFFINITY', pin_domain_affinity['affinity'])

        omp_num_threads = len([c for c in self.cpuinfo.pools_ondemand[0] if c.is_physical_core])
        if not args.logical_cores_for_ccl:
            omp_num_threads -= args.ccl_worker_count

        self.set_multi_thread_and_allocator(
                omp_num_threads,
                args.memory_allocator,
                False,
                args.omp_runtime,
                True,
                )

        os.environ['LAUNCH_CMD'] = '#'
        cmd = ['mpiexec.hydra']
        mpi_config = f'-l -np {args.nnodes * args.nprocs_per_node} -ppn {args.nprocs_per_node} -genv I_MPI_PIN_DOMAIN={pin_domain_affinity["pin_domain"]} -genv OMP_NUM_THREADS={omp_num_threads} '
        mpi_config += args.extra_mpi_params
        if args.nnodes > 1:
            mpi_config += f' -hostfile {args.hostfile}'
        cmd.extend(mpi_config.split())
        with_python = not args.no_python
        if with_python:
            cmd.append(sys.executable)
            cmd.append('-u')
        if args.module:
            cmd.append('-m')
        cmd.append(args.program)
        cmd.extend(args.program_args)
        log_name = f'{args.log_file_prefix}.log'
        log_name = os.path.join(args.log_path, log_name)
        cmd_s = ' '.join(cmd)
        if args.log_path:
            cmd_s = f'{cmd_s} 2>&1 | tee {log_name}'
        self.verbose('info', cmd_s)
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
        process.wait()
        os.environ['LAUNCH_CMD'] += f'{" ".join(cmd)},#'
        os.environ['LAUNCH_CMD'] = os.environ['LAUNCH_CMD'][:-2]

if __name__ == '__main__':
    pass
