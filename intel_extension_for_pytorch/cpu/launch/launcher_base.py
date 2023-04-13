import subprocess
import os
from os.path import expanduser
import glob
from .cpu_info import CPUPoolList

class Launcher():
    '''
    Base class for launcher
    '''
    def __init__(self, logger=None, lscpu_txt=''):
        self.logger = logger
        self.cpuinfo = CPUPoolList(self.logger, lscpu_txt)
        self.library_paths = []
        if "CONDA_PREFIX" in os.environ:
            self.library_paths.append(f'{os.environ["CONDA_PREFIX"]}/lib/')
        if "VIRTUAL_ENV" in os.environ:
            self.library_paths.append(f'{os.environ["VIRTUAL_ENV"]}/lib/')
        self.library_paths.extend([f'{expanduser("~")}/.local/lib/', '/usr/local/lib/',
                          '/usr/local/lib64/', '/usr/lib/', '/usr/lib64/',
                          '/usr/lib/x86_64-linux-gnu/'])
        self.ma_supported = ['auto', 'default', 'tcmalloc', 'jemalloc']
        self.omp_supported = ['auto', 'default', 'intel']

    def add_common_params(self, parser):
        group = parser.add_argument_group('Launcher Common Arguments')
        group.add_argument(
            '--memory-allocator',
            '--memory_allocator',
            default='auto',
            type=str,
            choices=self.ma_supported,
            help=f'Choose which memory allocator to run the workloads with. Supported choices are {self.ma_supported}.',
        )
        group.add_argument(
            '--omp-runtime',
            '--omp_runtime',
            default='auto',
            type=str,
            choices=self.omp_supported,
            help=f'Choose which OpenMP runtime to run the workloads with. Supported choices are {self.omp_supported}.',
        )

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

    def launch(self, args):
        pass

    def add_lib_preload(self, lib_type=None):
        '''
        Enable TCMalloc/JeMalloc/intel OpenMP
        '''
        lib_found = False
        lib_set = False
        for item in os.getenv('LD_PRELOAD', '').split(':'):
            if item.endswith(f'lib{lib_type}.so'):
                lib_set = True
                break
        if not lib_set:
            for lib_path in self.library_paths:
                if lib_path.endswith('/'):
                    lib_path = lib_path[:-1]
                library_file = f'{lib_path}/lib{lib_type}.so'
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    if 'LD_PRELOAD' in os.environ:
                        os.environ['LD_PRELOAD'] = f'{matches[0]}:{os.environ["LD_PRELOAD"]}'
                    else:
                        os.environ['LD_PRELOAD'] = matches[0]
                    lib_found = True
                    break
        return lib_set or lib_found

    def set_lib_bin_from_list(self, name_input, name_map, category, supported, fn, skip_list=[], extra_warning_msg_with_default_choice=''):
        '''
        Function to set libraries or commands that are predefined in support lists.
        The support list is formed in format ['auto', default choice, alternative A, alternative B, ...]. The list can only contain 'auto' and the default choice.
        Arguments:
            name_input: name of the lib/bin that user inputs.
            name_map: a dictionary. {'key': ['alias name', 'package installation command']} Its key is name of the lib/bin, its value is a list of string with 2 elements. First string of the list is alias name of the lib/bin that is searched in the system. For instance, when key is 'intel' for OpenMP runtime, the function will invoke fn (describe below) to search a library file 'libiomp5.so'. The fn function passed forms the library file name with its identifier 'iomp5'. Thus, the first string of this list for key 'intel' should be 'iomp5'. This value depends on how fn function searches for the lib/bin file. The second string should be a installation command guides users to install this package. When it is empty, the installation guide will not be prompted.
            category: category of this lib/bin. 'memory allocator', 'multi-task manager', etc.
            supported: predefined support list
            fn: a function how the lib/bin files will be searched. Return True to indicate a successful searching, otherwise return False.
            skip_list: a list containing name of lib/bin that will not be used.
            extra_warning_msg_with_default_choice: a warning message that will be prompted if designated choices are not available and fallen back to the default choice.
        '''
        name_local = name_input.lower()
        if not name_local in supported:
            name_local = supported[0]
            self.verbose('warning', f'Designated {category} \'{name_input}\' is unknown. Changing it to \'{name_local}\'. Supported {category} are {supported}.')
        if name_local in skip_list:
            name_local = supported[0]
            self.verbose('warning', f'Designated {category} \'{name_input}\' is not applicable at this moment. Changing it to \'{name_local}\'. Please choose another {category} from {supported}.')
        if name_local == supported[0]:
            for name in supported[2:]:
                if name in skip_list:
                    continue
                if fn(name_map[name][0]):
                    self.verbose('info', f'Use \'{name_local}\' => \'{name}\' {category}.')
                    name_local = name
                    break
            if name_local == supported[0]:
                name_local = supported[1]
                if len(supported[2:]) > 0:
                    msg = ''
                    if len(supported[2:]) == 1:
                        msg = f'\'{supported[2]}\' {category} is not found'
                    elif len(supported[2:]) < 3:
                        msg = f'Neither of {supported[2:]} {category} is found'
                    else:
                        msg = f'None of {supported[2:]} {category} is found'
                    self.verbose('warning', f'{msg} in {self.library_paths}.')
                if extra_warning_msg_with_default_choice != '':
                    extra_warning_msg_with_default_choice = f' {extra_warning_msg_with_default_choice}'
                self.verbose('info', f'Use \'{name_local}\' {category}.{extra_warning_msg_with_default_choice}')
        elif name_local in supported[2:]:
            if not fn(name_map[name_local][0]):
                extra_warning_msg_install_guide = ''
                if name_map[name_local][1] != '':
                    extra_warning_msg_install_guide = f' You can install it with "{name_map[name_local][1]}".'
                self.verbose('warning', f'Unable to find the \'{name_local}\' {category} library file in {self.library_paths}.{extra_warning_msg_install_guide}')
                name_local = supported[1]
                if extra_warning_msg_with_default_choice != '':
                    extra_warning_msg_with_default_choice = f' {extra_warning_msg_with_default_choice}'
                self.verbose('info', f'Use \'{name_local}\' {category}.{extra_warning_msg_with_default_choice}')
            else:
                self.verbose('info', f'Use \'{name_local}\' {category}.')
        else:
            self.verbose('info', f'Use \'{name_local}\' {category}.')
        return name_local

    def set_memory_allocator(self, memory_allocator='auto', benchmark=False):
        '''
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory resue and reduce page fault to improve performance.
        '''
        ma_lib_name = {
                'jemalloc': ['jemalloc', 'conda install -c conda-forge jemalloc'],
                'tcmalloc': ['tcmalloc', 'conda install -c conda-forge gperftools']
                }
        ma_local = self.set_lib_bin_from_list(memory_allocator, ma_lib_name, 'memory allocator', self.ma_supported, self.add_lib_preload, extra_warning_msg_with_default_choice='This may drop the performance.')
        if ma_local == 'jemalloc':
            if benchmark:
                self.set_env('MALLOC_CONF', 'oversize_threshold:1,background_thread:false,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1')
            else:
                self.set_env('MALLOC_CONF', 'oversize_threshold:1,background_thread:true,metadata_thp:auto')

    def set_omp_runtime(self, omp_runtime='auto', set_kmp_affinity=True):
        '''
        Set OpenMP runtime
        '''
        omp_lib_name = {'intel': ['iomp5', 'conda install intel-openmp']}
        omp_local = self.set_lib_bin_from_list(omp_runtime, omp_lib_name, 'OpenMP Runtime', self.omp_supported, self.add_lib_preload)
        if omp_local == 'intel':
            if set_kmp_affinity:
                self.set_env('KMP_AFFINITY', 'granularity=fine,compact,1,0')
            self.set_env('KMP_BLOCKTIME', '1')

    def logger_env(self, env_name=''):
        if env_name in os.environ:
            self.verbose('info', f'{env_name}={os.environ[env_name]}')

    def set_env(self, env_name, env_value=None):
        if not env_value:
            self.verbose('warning', f'{env_name} is None. Abandon setting environment variable {env_name}.')
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            self.verbose('warning', f'{env_name} in environment variable is {os.environ[env_name]} while the value you would like to set is {env_value}. Using the exsiting value.')
        self.logger_env(env_name)

    # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not. In scenario that use all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores. In this case, KMP_AFFINITY should not be set.
    def set_multi_thread_and_allocator(self, ncores_per_instance, memory_allocator='', benchmark=False, omp_runtime='', set_kmp_affinity=True):
        '''
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.
        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benifit.
        '''
        self.set_memory_allocator(memory_allocator, benchmark)
        self.set_omp_runtime(omp_runtime, set_kmp_affinity)
        self.logger_env('LD_PRELOAD')

if __name__ == '__main__':
    pass
