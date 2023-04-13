import itertools
import os
import platform
import re
import subprocess

# lscpu Examples
# # The following is the parsable format, which can be fed to other
# # programs. Each different item in every column has an unique ID
# # starting from zero.
# # CPU,Core,Socket,Node
# 0,0,0,0
# 1,0,0,0
# 2,1,0,0
# 3,1,0,0
# 4,2,1,1
# 5,2,1,1
# 6,3,1,1
# 7,3,1,1

# 0,0,0,
# 1,0,0,
# 2,1,0,
# 3,1,0,
# 4,2,0,
# 5,2,0,
# 6,3,0,
# 7,3,0,

# 0,0,0,0
# 1,1,0,0
# 2,2,1,1
# 3,3,1,1
# 4,0,0,0
# 5,1,0,0
# 6,2,1,1
# 7,3,1,1

class CoreInfo():
    '''
    Class to store core-specific information, including:
    - [int] CPU index
    - [int] Core index
    - [int] Numa node index
    - [bool] is a physical core or not
    '''
    def __init__(self, lscpu_txt=''):
        self.cpu = -1
        self.core = -1
        self.node = -1
        self.is_physical_core = True
        if lscpu_txt != '':
            self.parse_raw(lscpu_txt)

    def parse_raw(self, lscpu_txt):
        cols = lscpu_txt.split(',')
        idx_col_node = 3
        if cols[idx_col_node] == '':
            idx_col_node = 2
        self.cpu = int(cols[0])
        self.core = int(cols[1])
        self.node = int(cols[idx_col_node])

    def __str__(self):
        return f'{self.cpu}|{self.core}|{self.node}|{self.is_physical_core}'

class CPUPool(list):
    '''
    List of CoreInfo objects
    '''
    def __init__(self):
        super(CPUPool, self).__init__()

    def get_ranges(self, l):
        for a, b in itertools.groupby(enumerate(l), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    def get_pool_txt(self, return_mode='auto'):
        cpu_ids = [c.cpu for c in self]
        cpu_ranges = list(self.get_ranges(cpu_ids))
        cpu_ids_txt = ','.join([str(c) for c in cpu_ids])
        cpu_ranges_txt = ','.join([f'{r[0]}-{r[1]}' for r in cpu_ranges])
        node_ids_txt = ','.join([str(n) for n in sorted(list(set([c.node for c in self])))])
        ret = {'cores': '', 'nodes': node_ids_txt}
        if return_mode.lower() == 'list':
            ret['cores'] = cpu_ids_txt
        elif return_mode.lower() == 'range':
            ret['cores'] = cpu_ranges_txt
        else:
            if len(cpu_ids) <= len(cpu_ranges):
                ret['cores'] = cpu_ids_txt
            else:
                ret['cores'] = cpu_ranges_txt
        return ret

class CPUPoolList():
    '''
    Get a CPU pool with all available CPUs and CPU pools filtered with designated criterias.
    '''
    def __init__(self, logger=None, lscpu_txt=''):
        self.pool_all = CPUPool()
        self.pools_ondemand = []

        self.logger = logger
        if platform.system() == 'Windows':
            raise RuntimeError('Windows platform is not supported!!!')
        elif platform.system() == 'Linux':
            '''
            Retrieve CPU information from lscpu.
            '''
            if lscpu_txt == '':
                args = ['lscpu', '--parse=CPU,Core,Socket,Node']
                env_lang = os.getenv('LANG', 'UNSET')
                os.environ['LANG'] = 'C'
                lscpu_info = subprocess.check_output(args, env=os.environ, universal_newlines=True)
                if env_lang == 'UNSET':
                    del os.environ['LANG']
                else:
                    os.environ['LANG'] = env_lang
            else:
                lscpu_info = lscpu_txt

            '''
            Filter out lines that are really useful.
            '''
            lscpu_info = lscpu_info.strip().split('\n')
            for line in lscpu_info:
                line = line.strip()
                if re.match('^([\d]+,[\d]+,[\d]+,[\d]?)', line):
                    self.pool_all.append(CoreInfo(line))
            assert len(self.pool_all) > 0, 'cpuinfo is empty'

        '''
        Loop through all cores and determine is_physical_core for each of them.
        '''
        phy_cores = [c.core for c in self.pool_all]
        phy_cores_unique = set(phy_cores)
        assert len(phy_cores) % len(phy_cores_unique) == 0, f'Incorrect CPU configuration detected ({len(phy_cores_unique)} physical cores, {len(phy_cores)} logical cores).'
        if len(phy_cores) // len(phy_cores_unique) > 1:
            core_cur = -1
            self.pool_all.sort(key=lambda x: (x.core, x.cpu))
            for c in self.pool_all:
                if core_cur != c.core:
                    core_cur = c.core
                else:
                    c.is_physical_core = False
        self.pool_all.sort(key=lambda x: x.cpu)

    def verbose(self, level, msg):
        if self.logger:
            logging_fn = {
                    'warning': self.logger.warning,
                    'info': self.logger.info,
                    }
            assert level in logging_fn.keys(), f'Unrecognized logging level {level} is detected. Available levels are {logging_fn.keys()}.'
            logging_fn[level](msg)
        else:
            print(msg)

    '''
    Get CPU pools from all available CPU cores with designated criterias.
    - ninstances [int]: Number of instances. Should be a non negative integer, 0 by default. When it is 0, it will be set according to usage scenarios automatically in the function.
    - ncores_per_instance [int]: Number of cores per instance. Should be a non negative integer, 0 by default. When it is 0, it will be set according to usage scenarios automatically in the function.
    - use_logical_cores [bool]: Use logical cores on the workloads or not, False by default. When set to False, only physical cores are used.
    - skip_cross_node_cores [bool]: Allow instances to be executed on cores across NUMA nodes, False by default.
    - nodes_list [list]: A list containing all node ids that the execution is expected to be running on.
    - cores_list [list]: A list containing all cpu ids that the execution is expected to be running on.
    - return_mode [str]: A string that defines how result values are formed, could be either of 'auto', 'list' and 'range'. When set to 'list', a string with comma-separated cpu ids, '0,1,2,3,...', is returned. When set to 'range', a string with comma-separated cpu id ranges, '0-2,6-8,...', is returned. When set to 'auto', a 'list' or a 'range' whoever has less number of elements that are separated by comma is returned. I.e. for a list '0,1,2,6,7,8' and a range '0-2,6-8', both reflect the same cpu configuration, the range '0-2,6-8' is returned.
    '''
    def gen_pools_ondemand(
            self,
            ninstances=0,
            ncores_per_instance=0,
            use_logical_cores=False,
            skip_cross_node_cores=False,
            nodes_list=[],
            cores_list=[],
            return_mode='auto'):
        # Generate an aggregated CPU pool
        if len(cores_list) > 0:
            cores_available = [c.cpu for c in self.pool_all]
            assert set(cores_list).issubset(set(cores_available)), f'Designated cores list contains invalid cores.'
            if use_logical_cores:
                self.verbose('warning', 'Argument --use-logical-cores won\'t take effect when --cores-list is set.')
            pool = [c for c in self.pool_all if c.cpu in cores_list]
            nodes = list(set([c.node for c in pool]))
            ncores_per_node = -1
            for n in nodes:
                ncores_local = len([c for c in pool if c.node == n])
                if ncores_per_node == -1:
                    ncores_per_node = ncores_local
                else:
                    if ncores_per_node != ncores_local and skip_cross_node_cores:
                        skip_cross_node_cores = False
                        self.verbose('warning', 'Argument --skip-cross-node-cores cannot take effect on the designated cores. Disabled.')
                        break
        else:
            if len(nodes_list) > 0:
                nodes_available = set([c.node for c in self.pool_all])
                assert set(nodes_list).issubset(nodes_available), f'Designated nodes list contains invalid nodes.'
                pool = [c for c in self.pool_all if c.node in nodes_list]
            else:
                pool = self.pool_all
            if not use_logical_cores:
                pool = [c for c in pool if c.is_physical_core]

        # Determine ninstances and ncores_per_instance for grouping
        assert ncores_per_instance >= 0, 'Argument --ncores-per-instance cannot be a negative value.'
        assert ninstances >= 0, 'Argument --ninstances cannot be a negative value.'
        nodes = set([c.node for c in pool])
        if ncores_per_instance + ninstances == 0:
            # Both ncores_per_instance and ninstances are 0
            ninstances = 1
        if ncores_per_instance * ninstances == 0:
            # Either ncores_per_instance or ninstances is 0
            if skip_cross_node_cores:
                ncores_per_node = len(pool) // len(nodes)
                nresidual = 0
                if ncores_per_instance == 0:
                    nins_per_node = ninstances // len(nodes)
                    if ninstances % len(nodes) > 0:
                        nins_per_node += 1
                    ncores_per_instance = ncores_per_node // nins_per_node
                    nresidual = ncores_per_node % nins_per_node
                if ninstances == 0:
                    ninstances = ncores_per_node // ncores_per_instance * len(nodes)
                    nresidual = ncores_per_node % ncores_per_instance
                if nresidual > 0:
                    cores_remove = []
                    for n in nodes:
                        cores = [c for c in pool if c.node == n]
                        for i in range(nresidual):
                            cores_remove.append(cores[-1 * (i + 1)])
                    for c in cores_remove:
                        pool.remove(c)
            else:
                if ninstances == 0:
                    ninstances = len(pool) // ncores_per_instance
                if ncores_per_instance == 0:
                    ncores_per_instance = len(pool) // ninstances
        else:
            # Neither ncores_per_instance nor ninstances is 0
            if skip_cross_node_cores:
                self.verbose('warning', 'Argument --skip-cross-node-cores won\'t take effect when both --ninstances and --ncores-per-instance are explicitly set.')
        assert ninstances * ncores_per_instance > 0 and ninstances * ncores_per_instance <= len(pool), 'Requested number of cores exceeds what is available.'

        # Split the aggregated pool into individual pools
        self.pools_ondemand.clear()
        pool.sort(key=lambda x: (x.core, 1 - int(x.is_physical_core)))
        for i in range(ninstances):
            # Generate individual raw pool
            pool_local = CPUPool()
            for j in range(ncores_per_instance):
                pool_local.append(pool[i * ncores_per_instance + j])
            pool_local.sort(key=lambda x: x.cpu)
            self.pools_ondemand.append(pool_local)

if __name__ == "__main__":
    pools = CPUPoolList()
    pools.gen_pools_ondemand(use_logical_cores=False, nodes_list=[0,1], return_mode='auto', ninstances=3, ncores_per_instance=0, skip_cross_node_cores=False)
    print(f'capacity pool_auto:  {pools.pool_all.get_pool_txt(return_mode="auto")}')
    print(f'capacity pool_list:  {pools.pool_all.get_pool_txt(return_mode="list")}')
    print(f'capacity pool_range: {pools.pool_all.get_pool_txt(return_mode="range")}')
    print('')
    for i in range(len(pools.pools_ondemand)):
        p = pools.pools_ondemand[i]
        print(f'ondemand pool_auto:  {i} {p.get_pool_txt(return_mode="auto")}')
        print(f'ondemand pool_list:  {i} {p.get_pool_txt(return_mode="list")}')
        print(f'ondemand pool_range: {i} {p.get_pool_txt(return_mode="range")}')
        print([c.cpu for c in p])
