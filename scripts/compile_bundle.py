#!/usr/bin/env python
# encoding: utf-8

# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import urllib.parse

UTILSFILENAME = 'compilation_utils.py'
SYSTEM = platform.system()
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = SCRIPTDIR
SRCDIR = ''
if Path(BASEDIR).parts[-1] == 'scripts' and \
        os.path.isdir(os.path.join(BASEDIR, '..', 'intel_extension_for_pytorch')) and \
        os.path.exists(os.path.join(BASEDIR, '..', 'setup.py')):
    dir_parts = Path(BASEDIR).parts
    SRCDIR = dir_parts[-2]
    BASEDIR = os.path.join(*dir_parts[:-2])
if SRCDIR == '':
    SRCDIR = 'intel-extension-for-pytorch'

def _get_whl_from_dist(directory):
    whl_files = []
    for f in os.listdir(directory):
        if Path(f).suffix == '.whl':
            whl_files.append(f)
    assert len(whl_files) == 1, f'{len(whl_files)} files are found in {directory}, expect 1.'
    return os.path.join(directory, whl_files[0])

def _patch_libuv(dir,
                 pkg_name,
                 env,
                 show_command):
    path_uv = os.path.join(env['CONDA_PREFIX'], 'Library', 'bin', 'uv.dll')
    assert os.path.exists(path_uv)
    shutil.copy(path_uv, os.path.join(dir, pkg_name, 'lib'))

def _patchelf_so_files(dir,
                       pkg_name,
                       env,
                       show_command):
    exec_cmds(f'''find {pkg_name} -maxdepth 1 -name "*.so" -exec patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../../' --force-rpath {{}} \;
                  find {pkg_name}/lib -name "*.so" -exec patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../../' --force-rpath {{}} \;''',
              shell = True,
              cwd = dir,
              env = env,
              show_command = show_command)

def _patch_wheel(func,
                 *args):
    # args:
    #   directory
    #   func-specific args
    dir_dist = os.path.join(args[0], 'dist')
    dir_tmp = os.path.join(dir_dist, 'tmp')
    whl_file = _get_whl_from_dist(dir_dist)
    if os.path.isdir(dir_tmp):
        remove_directory(dir_tmp)
    os.mkdir(dir_tmp)
    import zipfile
    with zipfile.ZipFile(whl_file, "r") as zip_ref:
        zip_ref.extractall(dir_tmp)
    del sys.modules['zipfile']
    os.remove(whl_file)
    func_args = (dir_tmp,) + args[1:]
    func(*func_args)
    shutil.make_archive(whl_file, 'zip', dir_tmp)
    os.rename(f'{whl_file}.zip', whl_file)
    remove_directory(dir_tmp)

def _compile(directory,
             env,
             pkg_name = '',
             disable_oneapi_integration = False,
             incremental = False,
             show_command = False):
    print(f'========== {directory} ==========')
    dir_base = os.path.join(BASEDIR, directory)
    if not incremental:
        exec_cmds('python setup.py clean',
                  cwd = dir_base,
                  env = env,
                  show_command = show_command)
    dir_dist = os.path.join(dir_base, 'dist')
    if os.path.isdir(dir_dist):
        remove_directory(dir_dist)
    if show_command:
        print(env)
    redirect_file = os.path.join(dir_base, 'build.log')
    redirect_append = False
    if not env is None:
        if os.path.exists(redirect_file):
            os.remove(redirect_file)
        with open(redirect_file, 'w') as file:
            file.write('******************** Environment Variables ********************\n')
            for key, value in env.items():
                file.write(f'{key}: {value}\n')
            file.write('***************************************************************\n')
        redirect_append = True
    exec_cmds('python setup.py bdist_wheel',
              cwd = dir_base,
              env = env,
              redirect_file = redirect_file,
              redirect_append = redirect_append,
              show_command = show_command)
    if SYSTEM == 'Linux' and pkg_name != '' and not disable_oneapi_integration:
        _patch_wheel(_patchelf_so_files,
                     dir_base,
                     pkg_name,
                     env,
                     show_command)
    if SYSTEM == 'Windows' and pkg_name == 'torch':
        _patch_wheel(_patch_libuv,
                     dir_base,
                     pkg_name,
                     env,
                     show_command)
    exec_cmds(f'python -m pip install {_get_whl_from_dist(dir_dist)}',
              dir_base,
              env = env,
              show_command = show_command)

def process(*args):
    args_install_pytorch = args[0]
    args_abi = args[1]
    args_aot = args[2]
    args_max_jobs = args[3]
    args_ver_ipex = args[4]
    args_disable_oneapi_integration = args[5]
    args_with_vision = args[6]
    args_with_audio = args[7]
    args_with_torch_ccl = args[8]
    args_incremental = args[9]
    args_verbose = args[10]
    args_oneapi_root_dir = args[11]

    global exec_cmds
    global check_system_commands
    global remove_file_dir
    global remove_directory
    global clear_directory
    global update_source_code
    global source_env
    global get_duration
    global download
    utils_filepath = os.path.join(BASEDIR, SRCDIR, 'scripts', 'tools', 'compilation_helper', UTILSFILENAME) if BASEDIR != SCRIPTDIR else os.path.join(BASEDIR, UTILSFILENAME)
    import importlib.util
    spec = importlib.util.spec_from_file_location('script_module', utils_filepath)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules["script_module"] = utils_module
    spec.loader.exec_module(utils_module)
    exec_cmds = utils_module.exec_cmds
    check_system_commands = utils_module.check_system_commands
    remove_file_dir = utils_module.remove_file_dir
    remove_directory = utils_module.remove_directory
    clear_directory = utils_module.clear_directory
    update_source_code = utils_module.update_source_code
    source_env = utils_module.source_env
    get_duration = utils_module.get_duration
    download = utils_module.download

    n_cores = os.cpu_count()
    if args_max_jobs == 0:
        args_max_jobs = n_cores
    elif args_max_jobs > n_cores:
        print('--max-jobs argument is set a value larger than number of available cores. Reset it to the number of available cores.')
        args_max_jobs = n_cores
    else:
        pass

    # Check existence of required system commands
    commands = ['git']
    if args_with_vision:
        commands += ['conda']
    if SYSTEM == 'Linux':
        commands = ['gcc', 'g++']
    elif SYSTEM == 'Windows':
        if not 'conda' in commands:
            commands += ['conda']
    else:
        pass
    check_system_commands(commands)

    durations = {}

    # Update IPEX source code
    t0 = int(time.time() * 1000)
    update_source_code(SRCDIR,
                       'https://github.com/intel/intel-extension-for-pytorch.git',
                       args_ver_ipex,
                       basedir = BASEDIR,
                       show_command = args_verbose)
    durations['Retrieve IPEX source code'] = get_duration(t0)

    # Retrieve dependency information
    sys.path.append(os.path.join(BASEDIR, SRCDIR, 'scripts', 'tools',  'compilation_helper'))
    from dep_ver_utils import process_file as dep_info_retrieve
    INFO_TORCH = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), 'pytorch')
    INFO_TORCHVISION = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), 'torchvision')
    INFO_TORCHAUDIO = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), 'torchaudio')
    INFO_TORCHCCL = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), 'torch-ccl')
    VER_DPCPP = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), f'basekit:dpcpp-cpp-rt:{SYSTEM.lower()}')
    VER_MKL = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), f'basekit:mkl-dpcpp:{SYSTEM.lower()}')
    VER_CCL = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), f'basekit:oneccl-devel:{SYSTEM.lower()}')
    VER_MPI = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), f'basekit:impi-devel:{SYSTEM.lower()}')
    VER_PTI = dep_info_retrieve(os.path.join(BASEDIR, SRCDIR, 'dependency_version.json'), f'basekit:intel-pti:{SYSTEM.lower()}')
    if args_verbose:
        print(f'INFO_TORCH:       {str(INFO_TORCH)}')
        print(f'INFO_TORCHVISION: {str(INFO_TORCHVISION)}')
        print(f'INFO_TORCHAUDIO:  {str(INFO_TORCHAUDIO)}')
        print(f'INFO_TORCHCCL:    {str(INFO_TORCHCCL)}')
        print(f'VER_DPCPP:        {VER_DPCPP}')
        print(f'VER_MKL:          {VER_MKL}')
        print(f'VER_CCL:          {VER_CCL}')
        print(f'VER_MPI:          {VER_MPI}')
        print(f'VER_PTI:          {VER_PTI}')

    # Retrieve oneAPI information
    DPCPP_ROOT = ''
    OCLOC_ROOT = ''
    ONEMKL_ROOT = ''
    ONECCL_ROOT = ''
    MPI_ROOT = ''
    PTI_ROOT = ''

    count_threshold = 5
    if SYSTEM == 'Windows':
        count_threshold = 4
    if len(args_oneapi_root_dir) < count_threshold:
        DPCPP_ROOT = os.path.join(args_oneapi_root_dir[0], 'compiler', '.'.join(VER_DPCPP.split('.')[:2]))
        ONEMKL_ROOT = os.path.join(args_oneapi_root_dir[0], 'mkl', '.'.join(VER_MKL.split('.')[:2]))
        PTI_ROOT = os.path.join(args_oneapi_root_dir[0], 'pti', '.'.join(VER_PTI.split('.')[:2]))
        if SYSTEM == 'Linux':
            ONECCL_ROOT = os.path.join(args_oneapi_root_dir[0], 'ccl', '.'.join(VER_CCL.split('.')[:2]))
            MPI_ROOT = os.path.join(args_oneapi_root_dir[0], 'mpi', '.'.join(VER_MPI.split('.')[:2]))
        elif SYSTEM == 'Windows':
            OCLOC_ROOT = os.path.join(args_oneapi_root_dir[0], 'ocloc', '.'.join(VER_DPCPP.split('.')[:2]))
        else:
            pass
    else:
        DPCPP_ROOT = args_oneapi_root_dir[0]
        ONEMKL_ROOT = args_oneapi_root_dir[1]
        if SYSTEM == 'Linux':
            ONECCL_ROOT = args_oneapi_root_dir[2]
            MPI_ROOT = args_oneapi_root_dir[3]
            PTI_ROOT = args_oneapi_root_dir[4]
        elif SYSTEM == 'Windows':
            PTI_ROOT = args_oneapi_root_dir[2]
            OCLOC_ROOT = args_oneapi_root_dir[3]
        else:
            pass

    # Verify existence of oneAPI environments
    DPCPP_ENV = ''
    OCLOC_ENV = ''
    ONEMKL_ENV = ''
    PTI_ENV = ''
    ONECCL_ENV = ''
    MPI_ENV = ''

    if SYSTEM == 'Linux':
        DPCPP_ENV = os.path.join(DPCPP_ROOT, 'env', 'vars.sh')
        ONEMKL_ENV = os.path.join(ONEMKL_ROOT, 'env', 'vars.sh')
        PTI_ENV = os.path.join(PTI_ROOT, 'env', 'vars.sh')
        ONECCL_ENV = os.path.join(ONECCL_ROOT, 'env', 'vars.sh')
        MPI_ENV = os.path.join(MPI_ROOT, 'env', 'vars.sh')
    elif SYSTEM == 'Windows':
        DPCPP_ENV = os.path.join(DPCPP_ROOT, 'env', 'vars.bat')
        OCLOC_ENV = os.path.join(OCLOC_ROOT, 'env', 'vars.bat')
        ONEMKL_ENV = os.path.join(ONEMKL_ROOT, 'env', 'vars.bat')
        PTI_ENV = os.path.join(PTI_ROOT, 'env', 'vars.bat')
    else:
        pass

    assert os.path.exists(DPCPP_ENV), f'DPC++ compiler environment {DPCPP_ROOT} doesn\'t exist.'
    assert os.path.exists(ONEMKL_ENV), f'oneMKL environment {ONEMKL_ROOT} doesn\'t exist.'
    assert os.path.exists(PTI_ENV), f'PTI environment {PTI_ROOT} doesn\'t exist.'
    if ONECCL_ENV != '':
        assert os.path.exists(ONECCL_ENV), f'oneCCL environment {ONECCL_ROOT} doesn\'t exist.'
    if MPI_ENV != '':
        assert os.path.exists(MPI_ENV), f'Intel(R) MPI environment {MPI_ROOT} doesn\'t exist.'
    if OCLOC_ENV != '':
        assert os.path.exists(OCLOC_ENV), f'OpenCL™ Offline Compiler (OCLOC) environment {OCLOC_ROOT} doesn\'t exist.'

    if args_verbose:
        print(f'DPCPP_ENV:  {DPCPP_ENV}')
        print(f'ONEMKL_ENV: {ONEMKL_ENV}')
        print(f'PTI_ENV:    {PTI_ENV}')
        print(f'ONECCL_ENV: {ONECCL_ENV}')
        print(f'MPI_ENV:    {MPI_ENV}')
        print(f'OCLOC_ENV:  {OCLOC_ENV}')

    # Adjust arguments to avoid conflict configurations
    # install_mode: Indicate how to install PyTorch packages
    #               skip:    no touch
    #               pip:     install the prebuilt wheel files
    #               compile: compile from source
    install_mode = {'torch': '',
                    'vision': '',
                    'audio': ''}
    if args_install_pytorch == '':
        r, _ = exec_cmds('python -c "import torch"',
                         silent = True,
                         shell = True,
                         exit_on_failure = False,
                         stop_on_failure = False)
        if r == 0:
            install_mode['torch'] = 'skip'
            if INFO_TORCHVISION['commit'] == 'N/A':
                args_with_vision = False
            if INFO_TORCHAUDIO['commit'] == 'N/A':
                args_with_audio = False
            if args_with_vision:
                install_mode['vision'] = 'compile'
            if args_with_audio:
                install_mode['audio'] = 'compile'
        else:
            install_mode['torch'] = 'pip'
            if INFO_TORCHVISION['version'] == 'N/A':
                args_with_vision = False
            if INFO_TORCHAUDIO['version'] == 'N/A':
                args_with_audio = False
            if args_with_vision:
                install_mode['vision'] = 'pip'
            if args_with_audio:
                install_mode['audio'] = 'pip'
    elif args_install_pytorch == 'pip':
        install_mode['torch'] = 'pip'
        if INFO_TORCHVISION['version'] == 'N/A':
            args_with_vision = False
        if INFO_TORCHAUDIO['version'] == 'N/A':
            args_with_audio = False
        if args_with_vision:
            install_mode['vision'] = 'pip'
        if args_with_audio:
            install_mode['audio'] = 'pip'
    elif args_install_pytorch == 'compile':
        install_mode['torch'] = 'compile'
        if INFO_TORCHVISION['commit'] == 'N/A':
            args_with_vision = False
        if INFO_TORCHAUDIO['commit'] == 'N/A':
            args_with_audio = False
        if args_with_vision:
            install_mode['vision'] = 'compile'
        if args_with_audio:
            install_mode['audio'] = 'compile'
    else:
        pass
    if SYSTEM != 'Linux':
        args_with_torch_ccl = False
    else:
        if INFO_TORCHCCL['commit'] == 'N/A':
            args_with_torch_ccl = False

    if install_mode['torch'] == 'compile':
        update_source_code('pytorch',
                           'https://github.com/pytorch/pytorch.git',
                           INFO_TORCH['commit'],
                           basedir = BASEDIR,
                           show_command = args_verbose)
    if args_with_vision and install_mode['vision'] == 'compile':
        update_source_code('vision',
                           'https://github.com/pytorch/vision.git',
                           INFO_TORCHVISION['commit'],
                           basedir = BASEDIR,
                           show_command = args_verbose)
    if args_with_audio and install_mode['audio'] == 'compile':
        update_source_code('audio',
                           'https://github.com/pytorch/audio.git',
                           INFO_TORCHAUDIO['commit'],
                           basedir = BASEDIR,
                           show_command = args_verbose)
    if args_with_torch_ccl:
        update_source_code('torch-ccl',
                           'https://github.com/intel/torch-ccl.git',
                           INFO_TORCHCCL['commit'],
                           branch_main = 'master',
                           basedir = BASEDIR,
                           show_command = args_verbose)

    # Clean Python environment
    t0 = int(time.time() * 1000)
    if install_mode['torch'] != 'skip':
        exec_cmds('python -m pip uninstall -y torch pytorch-triton-xpu torchvision torchaudio',
                  show_command = args_verbose)
    exec_cmds('''python -m pip uninstall -y intel-extension-for-pytorch intel-extension-for-pytorch-deepspeed oneccl_bind_pt
                 python -m pip install "cmake<4" make ninja requests''',
              shell = True,
              show_command = args_verbose)
    durations['Clean Python environment'] = get_duration(t0)

    # Prepare patchelf env on Linux
    if SYSTEM == 'Linux':
        import tarfile
        t0 = int(time.time() * 1000)
        file = os.path.join(BASEDIR, 'patchelf.tar.gz')
        directory = os.path.join(BASEDIR, 'patchelf')
        download('https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz', file)
        if os.path.isdir(directory):
            remove_directory(directory)
        with tarfile.open(file, 'r:gz') as tar:
            tar.extractall(path = directory)
        os.remove(file)
        durations['Prepare patchelf environment'] = get_duration(t0)
        del sys.modules['tarfile']
    elif SYSTEM == 'Windows':
        if install_mode['torch'] == 'compile':
            import zipfile
            t0 = int(time.time() * 1000)
            file = os.path.join(BASEDIR, 'level_zero_sdk.zip')
            directory = os.path.join(BASEDIR, 'level_zero_sdk')
            download('https://github.com/oneapi-src/level-zero/releases/download/v1.14.0/level-zero-sdk_1.14.0.zip', file)
            if os.path.isdir(directory):
                remove_directory(directory)
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(directory)
            os.remove(file)
            durations['Prepare Level-Zero SDK environment'] = get_duration(t0)
            del sys.modules['zipfile']

    # Prepare compilation environment
    t0 = int(time.time() * 1000)
    env = os.environ.copy()
    env = source_env(DPCPP_ENV, env, show_command = args_verbose)
    env = source_env(PTI_ENV, env, show_command = args_verbose)
    if SYSTEM == 'Linux':
        env['PATH'] = f'{os.path.join(BASEDIR, "patchelf", "bin")}{os.pathsep}{env["PATH"]}'
    elif SYSTEM == 'Windows':
        env = source_env(OCLOC_ENV, env, show_command = args_verbose)
        env['DISTUTILS_USE_SDK'] = '1'
    else:
        pass
    env['MAX_JOBS'] = str(args_max_jobs)
    durations['Prepare compilation environment'] = get_duration(t0)

    # Install PyTorch
    if install_mode['torch'] == 'compile':
        t0 = int(time.time() * 1000)
        exec_cmds('python -m pip install -r requirements.txt',
                  cwd = os.path.join(BASEDIR, 'pytorch'),
                  show_command = args_verbose)
        if SYSTEM == 'Linux':
            exec_cmds('python -m pip install --force-reinstall mkl-static mkl-include',
                      show_command = args_verbose)
        elif SYSTEM == 'Windows':
            exec_cmds('conda install -y libuv -c conda-forge',
                      shell = True,
                      show_command = args_verbose)
        else:
            pass
        env_torch = env.copy()
        # env_torch['PYTORCH_BUILD_VERSION'] = INFO_TORCH['version']
        # env_torch['PYTORCH_BUILD_NUMBER'] = '0'
        if 'CONDA_PREFIX' in env_torch:
            if 'CMAKE_PREFIX_PATH' in env_torch:
                env_torch['CMAKE_PREFIX_PATH'] = f'{env_torch["CONDA_PREFIX"]}{os.pathsep}{env_torch["CMAKE_PREFIX_PATH"]}'
            else:
                env_torch['CMAKE_PREFIX_PATH'] = env_torch['CONDA_PREFIX']
        if 'VIRTUAL_ENV' in env_torch:
            if 'CMAKE_PREFIX_PATH' in env_torch:
                env_torch['CMAKE_PREFIX_PATH'] = f'{env_torch["VIRTUAL_ENV"]}{os.pathsep}{env_torch["CMAKE_PREFIX_PATH"]}'
            else:
                env_torch['CMAKE_PREFIX_PATH'] = env_torch['VIRTUAL_ENV']
        env_torch['USE_NUMA'] = '0'
        env_torch['USE_CUDA'] = '0'
        env_torch['USE_MPI'] = '0'
        aot = args_aot
        if aot == '':
            # https://github.com/intel/torch-xpu-ops/blob/release/2.7/cmake/BuildFlags.cmake#L120-L124
            if SYSTEM == 'Linux':
                aot = 'pvc,bmg,dg2,arl-h,mtl-h,lnl-m'
            elif SYSTEM == 'Windows':
                aot = 'mtl,mtl-h,bmg,dg2,arl-h,lnl-m'
            else:
                aot = 'none'
        env_torch['TORCH_XPU_ARCH_LIST'] = aot
        if SYSTEM == 'Linux':
            env_torch['USE_STATIC_MKL'] = '1'
            env_torch['_GLIBCXX_USE_CXX11_ABI'] = str(args_abi)
        elif SYSTEM == 'Windows':
            env_torch['XPU_ENABLE_KINETO'] = '1'
            env_torch['INCLUDE'] = f'{os.path.join(BASEDIR, "level_zero_sdk", "include")}{os.pathsep}{env_torch["INCLUDE"]}'
        else:
            pass
        if not args_disable_oneapi_integration:
            # https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py
            env_torch['PYTORCH_EXTRA_INSTALL_REQUIREMENTS'] = 'intel-cmplr-lib-rt==2025.0.4; platform_system == "Linux" | intel-cmplr-lib-ur==2025.0.4; platform_system == "Linux" | intel-cmplr-lic-rt==2025.0.4; platform_system == "Linux" | intel-sycl-rt==2025.0.4; platform_system == "Linux" | intel-cmplr-lib-rt==2025.0.5; platform_system == "Windows" | intel-cmplr-lib-ur==2025.0.5; platform_system == "Windows" | intel-cmplr-lic-rt==2025.0.5; platform_system == "Windows" | intel-sycl-rt==2025.0.5; platform_system == "Windows" | tcmlib==1.2.0 | umf==0.9.1 | intel-pti==0.10.1'
        _compile('pytorch',
                 env_torch,
                 pkg_name = 'torch',
                 disable_oneapi_integration = args_disable_oneapi_integration,
                 show_command = args_verbose)
        if SYSTEM == 'Windows':
            remove_directory(os.path.join(BASEDIR, 'level_zero_sdk'))
        ver_triton = '=='
        with open(os.path.join(BASEDIR, 'pytorch', '.ci', 'docker', 'triton_version.txt'), 'r') as file:
            ver_triton += file.read().strip()
        if ver_triton == '==':
            ver_triton = ''
        exec_cmds(f'python -m pip install pytorch-triton-xpu{ver_triton} --index-url {INFO_TORCH["index-url"]}',
                  show_command = args_verbose)
        if SYSTEM == 'Linux':
            exec_cmds('python -m pip uninstall -y mkl-static mkl-include',
                      show_command = args_verbose)
        durations['Compile PyTorch'] = get_duration(t0)
    elif install_mode['torch'] == 'pip':
        t0 = int(time.time() * 1000)
        command = f'python -m pip install torch=={INFO_TORCH["version"]}'
        if args_with_vision:
            command += f' torchvision=={INFO_TORCHVISION["version"]}'
        if args_with_audio:
            command += f' torchaudio=={INFO_TORCHAUDIO["version"]}'
        command += f' --index-url {INFO_TORCH["index-url"]}'
        exec_cmds(command,
                  show_command = args_verbose)
        durations['Install PyTorch packages'] = get_duration(t0)
    else:
        pass

    t0 = int(time.time() * 1000)
    _, lines_stdout = exec_cmds('python -c "import torch; print(\\",\\".join(torch._C._xpu_getArchFlags().split()));"',
                                silent = True,
                                shell = True)
    if args_aot == '':
        args_aot = lines_stdout[len(lines_stdout) - 1]
    durations['Prepare compilation environment'] += get_duration(t0)

    # Install TorchVision
    if args_with_vision and install_mode['vision'] == 'compile':
        t0 = int(time.time() * 1000)
        env_vision = env.copy()
        exec_cmds('''python -m pip install Pillow
                     conda install -y libpng libjpeg-turbo -c conda-forge''',
                  shell = True,
                  show_command = args_verbose)
        _compile('vision',
                 env_vision,
                 show_command = args_verbose)
        durations['Compile TorchVision'] = get_duration(t0)

    # Install TorchAudio
    if args_with_audio and install_mode['audio'] == 'compile':
        t0 = int(time.time() * 1000)
        env_audio = env.copy()
        exec_cmds('python -m pip install -r requirements.txt',
                  cwd = os.path.join(BASEDIR, 'audio'),
                  show_command = args_verbose)
        _compile('audio',
                 env_audio,
                 show_command = args_verbose)
        durations['Compile TorchAudio'] = get_duration(t0)

    # Install Intel® Extension for PyTorch*
    t0 = int(time.time() * 1000)
    env = source_env(ONEMKL_ENV, env, show_command = args_verbose)
    durations['Prepare compilation environment'] += get_duration(t0)

    t0 = int(time.time() * 1000)
    env_ipex = env.copy()
    exec_cmds('python -m pip install -r requirements.txt',
              cwd = os.path.join(BASEDIR, SRCDIR),
              show_command = args_verbose)
    env_ipex['BUILD_WITH_CPU'] = '0'
    env_ipex['TORCH_XPU_ARCH_LIST'] = args_aot
    env_ipex['ENABLE_ONEAPI_INTEGRATION'] = str(int(not args_disable_oneapi_integration))
    _compile(SRCDIR,
             env_ipex,
             pkg_name = 'intel_extension_for_pytorch',
             disable_oneapi_integration = args_disable_oneapi_integration,
             incremental = args_incremental,
             show_command = args_verbose)
    durations['Compile IPEX'] = get_duration(t0)

    # Install Torch-ccl
    if args_with_torch_ccl:
        t0 = int(time.time() * 1000)
        if SYSTEM == 'Linux':
            env = source_env(ONECCL_ENV, env, show_command = args_verbose)
            env = source_env(MPI_ENV, env, show_command = args_verbose)
            env['ONEAPIROOT'] = os.path.join(ONECCL_ROOT, '..', '..')
        durations['Prepare compilation environment'] += get_duration(t0)

        t0 = int(time.time() * 1000)
        env_torchccl = env.copy()
        env_torchccl['USE_SYSTEM_ONECCL'] = '1'
        env_torchccl['INTELONEAPIROOT'] = env_torchccl['ONEAPIROOT']
        env_torchccl['COMPUTE_BACKEND'] = 'dpcpp'
        _compile('torch-ccl',
                 env_torchccl,
                 pkg_patchelf = 'oneccl_bindings_for_pytorch',
                 disable_oneapi_integration = args_disable_oneapi_integration,
                 show_command = args_verbose)
        durations['Compile Torch-CCL'] = get_duration(t0)

    for directory in [os.path.join(BASEDIR, 'patchelf'),
                      os.path.join(BASEDIR, 'level_zero_sdk')]:
        if os.path.isdir(directory):
            remove_directory(directory)

    # Print step duration
    print('')
    print('******************** Compilation Finished ********************')
    for key in sorted(durations):
        print(f'{key}: {durations[key]:.2f}s')
    print('')

    # Sanity Test
    print('************************* Sanity Test ************************')
    if SYSTEM == 'Linux':
        _, libstdcpp = exec_cmds(f'bash ./{SRCDIR}/scripts/tools/compilation_helper/get_libstdcpp_lib.sh',
                                 cwd = BASEDIR,
                                 silent = True,
                                 show_command = args_verbose)
        assert len(libstdcpp) == 1, 'Something goes wrong when finding libstdcpp'
        if not libstdcpp[0].startswith('/usr/lib/'):
            os.environ['LD_PRELOAD'] = libstdcpp[0]
            print(f'Note: Set environment variable "export LD_PRELOAD={libstdcpp[0]}" to avoid the "version `GLIBCXX_N.N.NN\' not found" error.')
            print('')

    import torch
    print(f'torch_version:       {torch.__version__}')
    print(f'torch_cxx11_abi:     {str(int(torch._C._GLIBCXX_USE_CXX11_ABI))}')
    print(f'torch_aot:           {",".join(torch._C._xpu_getArchFlags().split())}')
    if args_with_vision:
        import torchvision
        print(f'torchvision_version: {torchvision.__version__}')
    if args_with_audio:
        import torchaudio
        print(f'torchaudio_version:  {torchaudio.__version__}')
    import intel_extension_for_pytorch as ipex
    print(f'ipex_version:        {ipex.__version__}')
    print(f'ipex_aot:            {ipex.__build_aot__}')
    if args_with_torch_ccl:
        import oneccl_bindings_for_pytorch as torch_ccl
        print(f'torchccl_version:    {torch_ccl.__version__}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to simply setting up Intel® Extension for PyTorch* environment, including installation/compilation of PyTorch and/or TorchVision/TorchAudio/Torch-CCL.')
    parser.add_argument(
        'oneapi_root_dir',
        nargs = '+',
        help = 'Root directory of oneAPI components.',
        type = str,
    )
    parser.add_argument(
        '--max-jobs',
        help = 'Number of cores used for the compilation. Setting it to 0 for automatically detection. Value is 0 by default.',
        type = int,
        default = 0,
    )
    parser.add_argument(
        '--abi',
        help = 'Set value for _GLIBCXX_USE_CXX11_ABI if PyTorch is compiled from source on Linux. Value is 1 by default.',
        type = int,
        choices = [0, 1],
        default = 1,
    )
    parser.add_argument(
        '--install-pytorch',
        help = 'Indicate how to install PyTorch. Can be "pip" for installing the prebuilt wheel file, and "compile" for compiling from source.',
        type = str,
        choices = ['', 'pip', 'compile'],
        default = '',
    )
    parser.add_argument(
        '--aot',
        help = 'AOT text for Ahead-Of-Time compilation. Setting it to "none" performs compilation without AOT.',
        type = str,
        default = '',
    )
    parser.add_argument(
        '--ver-ipex',
        help = 'Designate a specific branch/tag of Intel® Extension for PyTorch* source code to build with.',
        type = str,
        default = '',
        )
    parser.add_argument(
        '--disable-oneapi-integration',
        help = 'Avoid integrating oneAPI components as dependency packages.',
        action = 'store_true',
    )
    parser.add_argument(
        '--with-vision',
        help = 'Install TorchVision.',
        action = 'store_true',
    )
    parser.add_argument(
        '--with-audio',
        help = 'Install TorchAudio.',
        action = 'store_true',
    )
    parser.add_argument(
        '--with-torch-ccl',
        help = 'Install Torch-CCL.',
        action = 'store_true',
    )
    parser.add_argument(
        '--incremental',
        help = 'Enable IPEX incremental compilation.',
        action = 'store_true',
    )
    parser.add_argument(
        '--verbose',
        help = 'Show more information for debugging compilation.',
        action = 'store_true',
    )
    args = parser.parse_args()

    utils_filepath = os.path.join(BASEDIR, UTILSFILENAME)
    if BASEDIR != SCRIPTDIR:
        assert args.ver_ipex == '', 'Argument --ver-ipex cannot be set if you run the script from a exisiting source code directory.'
    else:
        assert args.ver_ipex != '', 'Argument --ver-ipex must be set to a branch/tag/commit of Intel® Extension for PyTorch* source code.'
        if os.path.isfile(utils_filepath):
            os.remove(utils_filepath)
        url = f'https://github.com/intel/intel-extension-for-pytorch/blob/{urllib.parse.quote(args.ver_ipex)}/scripts/tools/compilation_helper/{UTILSFILENAME}'
        import subprocess
        p = subprocess.Popen('python -m pip install requests',
                     stdout = subprocess.PIPE,
                     stderr = subprocess.STDOUT,
                     shell = True,
                     text = True)
        for line in iter(p.stdout.readline, ''):
            pass
        import requests
        response = requests.get(url)
        assert response.status_code >= 200 and response.status_code < 300, f'Failed to access {url}, status code: {response.status_code}.'
        urls = re.findall('"rawBlobUrl":"(.*?)"', response.text)
        assert len(urls) == 1, f'Unexpected number of raw URLs retrieved.\n{matches}'
        response = requests.get(urls[0], stream=True)
        with open(utils_filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    process(
            args.install_pytorch,
            args.abi,
            args.aot,
            args.max_jobs,
            args.ver_ipex,
            args.disable_oneapi_integration,
            args.with_vision,
            args.with_audio,
            args.with_torch_ccl,
            args.incremental,
            args.verbose,
            args.oneapi_root_dir
    )

    for item in [os.path.join(BASEDIR, '__pycache__'),
                 utils_filepath]:
        remove_file_dir(item)
