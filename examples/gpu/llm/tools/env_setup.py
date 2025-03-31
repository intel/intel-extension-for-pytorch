#!/usr/bin/env python
# encoding: utf-8

# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from compilation_utils import exec_cmds, remove_directory, remove_file_dir

SYSTEM = platform.system()
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(*Path(SCRIPTDIR).parts[:-1])
WHEELDIR = os.path.join(BASEDIR, 'wheels')
SUFFIX = '.sh' if SYSTEM == 'Linux' else '.bat' if SYSTEM == 'Windows' else ''
AUX_INSTALL_SCRIPT = os.path.join(WHEELDIR, f'aux_install{SUFFIX}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to simply setting up LLM environment with Intel® Extension for PyTorch*, including installation/compilation of PyTorch and Torch-CCL.')

    # Define arguments
    parser.add_argument(
        '--setup',
        help = 'Setup the LLM environment.',
        action = 'store_true',
    )
    parser.add_argument(
        '--deploy',
        help = 'Deploy the LLM environment.',
        action = 'store_true',
    )
    parser.add_argument(
        '--oneapi-root-dir',
        nargs = '+',
        help = 'Root directory of oneAPI components.',
        type = str,
        default = '',
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
        '--verbose',
        help = 'Show more information for debugging compilation.',
        action = 'store_true',
    )

    # Parse arguments
    args = parser.parse_args()

    if args.setup:
        SRCDIR = os.path.join(*Path(BASEDIR).parts[:-3])
        assert os.path.exists(os.path.join(SRCDIR, 'dependency_version.json')), f'Please check if {SRCDIR} is a valid Intel® Extension for PyTorch* source code directory.'

        sys.path.append(os.path.join(SRCDIR, 'scripts', 'tools',  'compilation_helper'))
        from dep_ver_utils import process_file as dep_info_retrieve
        INFO_TORCH = dep_info_retrieve(os.path.join(SRCDIR, 'dependency_version.json'), 'pytorch')
        INFO_TORCHCCL = dep_info_retrieve(os.path.join(SRCDIR, 'dependency_version.json'), 'torch-ccl')

        VER_IPEX_MAJOR = ''
        VER_IPEX_MINOR = ''
        VER_IPEX_PATCH = ''
        with open(os.path.join(SRCDIR, 'version.txt'), 'r') as file:
            while True:
                ln = file.readline().strip()
                if not ln:
                    break
                lst = ln.split(' ')
                if lst[0] == 'VERSION_MAJOR':
                    VER_IPEX_MAJOR = lst[1]
                if lst[0] == 'VERSION_MINOR':
                    VER_IPEX_MINOR = lst[1]
                if lst[0] == 'VERSION_PATCH':
                    VER_IPEX_PATCH = lst[1]
        assert VER_IPEX_MAJOR != '' and VER_IPEX_MINOR != '' and VER_IPEX_PATCH != ''
        VER_IPEX=f'{VER_IPEX_MAJOR}.{VER_IPEX_MINOR}.{VER_IPEX_PATCH}+xpu'

        if os.path.isdir(WHEELDIR):
            remove_directory(WHEELDIR)
        os.mkdir(WHEELDIR)

        cont_aux = []
        if SYSTEM == 'Linux':
            cont_aux.append('#!/bin/bash')
            cont_aux.append('set -e')
        elif SYSTEM == 'Windows':
            cont_aux.append('@echo off')
        else:
            pass
        if len(args.oneapi_root_dir) == 0:
            torchccl = ''
            if SYSTEM == 'Linux':
                torchccl = f'oneccl-bind-pt=={INFO_TORCHCCL["version"]}'
            cont_aux.append(f'python -m pip install torch=={INFO_TORCH["version"]} --index-url {INFO_TORCH["index-url"]}')
            cont_aux.append(f'python -m pip install intel-extension-for-pytorch=={VER_IPEX} {torchccl} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/')
        else:
            sys.path.append(os.path.join(SRCDIR, 'scripts'))
            from compile_bundle import process as binary_compile
            binary_compile(
                args.install_pytorch,
                args.abi,
                args.aot,
                args.max_jobs,
                '',
                False,
                False,
                False,
                True,
                False,
                args.verbose,
                args.oneapi_root_dir
            )
            COMPILEDIR = os.path.join(*Path(SRCDIR).parts[:-1])
            for root, _, files in os.walk(COMPILEDIR, followlinks=True):
                dir_parts = Path(root).parts
                dir_name = dir_parts[-1] if dir_parts else root
                if dir_name != 'dist':
                    continue
                for file in files:
                    if Path(file).suffix == '.whl':
                        shutil.copy(os.path.join(root, file), WHEELDIR)
            for item in ['llvm-project', 'llvm-release', 'torch-ccl']:
                remove_file_dir(os.path.join(COMPILEDIR, item))
            cmd_triton = ''
            pytorch_dir = os.path.join(COMPILEDIR, 'pytorch')
            if os.path.isdir(pytorch_dir):
                ver_triton = '=='
                with open(os.path.join(pytorch_dir, '.ci', 'docker', 'triton_version.txt'), 'r') as file:
                    ver_triton += file.read().strip()
                if ver_triton == '==':
                    ver_triton = ''
                cmd_triton = f'python -m pip install pytorch-triton-xpu{ver_triton} --index-url {INFO_TORCH["index-url"]}'
                remove_directory(pytorch_dir)
            else:
                cont_aux.append(f'python -m pip install torch=={INFO_TORCH["version"]} --index-url {INFO_TORCH["index-url"]}')
            if SYSTEM == 'Windows':
                cont_aux.append(f'for %%f in ({os.path.join(".", "wheels", "*.whl")}) do python -m pip install "%%f"')
            else:
                cont_aux.append(f'python -m pip install {os.path.join(".", "wheels", "*.whl")}')
            if cmd_triton != '':
                cont_aux.append(cmd_triton)
        cont_aux.append('python -m pip install -r requirements.txt')
        with open(AUX_INSTALL_SCRIPT, 'w') as file:
            file.write('\n'.join(cont_aux))

    if args.deploy:
        env = os.environ.copy()
        if 'LIBRARY_PATH' in env and 'CONDA_PREFIX' in env:
            env['LIBRARY_PATH'] = f'{env["CONDA_PREFIX"]}/lib{os.pathsep}{env["CONDA_PREFIX"]}'
        assert os.path.exists(AUX_INSTALL_SCRIPT), f'{AUX_INSTALL_SCRIPT} doesn\'t exist.'
        launch_cmd = 'call' if SYSTEM == 'Windows' else 'bash'
        exec_cmds(f'{launch_cmd} {AUX_INSTALL_SCRIPT}',
                   cwd = BASEDIR,
                   env = env,
                   shell = True,
                   show_command = args.verbose)
        remove_directory(WHEELDIR)
