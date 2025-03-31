#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from compilation_utils import exec_cmds, check_system_commands

SYSTEM = platform.system()
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(*Path(SCRIPTDIR).parts[:-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to simply setting up LLM environment.')

    # Define arguments
    parser.add_argument(
        'directory',
        help = 'Directory of examples to run.',
        type = str,
        choices = ['inference', 'fine-tuning', 'bitsandbytes', 'training'],
    )

    # Parse arguments
    args = parser.parse_args()

    exec_dir = os.path.join(BASEDIR, args.directory)
    env = os.environ.copy()

    dict_cmds = check_system_commands(['git'])
    if SYSTEM != 'Windows':
        check_system_commands(['patch'])
    else:
        dir_git_bin = os.path.join(*Path(dict_cmds['git']).parts[:-2])
        dir_patch_bin = os.path.join(dir_git_bin, 'usr', 'bin')
        assert shutil.which(os.path.join(dir_patch_bin, 'patch.exe')), 'Command patch is not found.'
        if not dir_patch_bin in env['PATH'].split(os.pathsep):
            env['PATH'] = f'{dir_patch_bin}{os.pathsep}{env["PATH"]}'

    exec_cmds('python -m pip install -r requirements.txt',
              cwd = exec_dir,
              show_command = True,
              shell = True)

    file_patch = os.path.join(exec_dir, 'patches', 'transformers.patch')
    if os.path.isfile(file_patch):
        _, lns = exec_cmds('python -c "import transformers; print(transformers.__path__[0]);"',
                           silent = True,
                           shell = True)
        pkgdir = ''
        for ln in lns:
            if not ln.startswith('['):
                pkgdir = ln
        if pkgdir == '':
            print('Transformers not found. Skipping performance metrics patching...')
        else:
            pattern_found = False
            regex = re.compile('token_latency')
            for root, _, files in os.walk(pkgdir):
                if pattern_found:
                    break
                for file in files:
                    if pattern_found:
                        break
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', errors='ignore') as f:
                        for line in f:
                            if regex.search(line):
                                pattern_found = True
                                break
            if pattern_found:
                print('The transformers package is already patched. Skip patching...')
            else:
                print('Patching Transformers for performance metrics enabling...')
                r, _ = exec_cmds(f'patch -d {pkgdir} -p3 -t < {file_patch}',
                                 cwd = exec_dir,
                                 env = env,
                                 shell = True,
                                 exit_on_failure = False)
                if r > 0:
                    print('Patching failed.')
                    for root, _, files in os.walk(pkgdir):
                        for file in files:
                            path_file = Path(file)
                            if path_file.suffix == '.rej':
                                os.remove(os.path.join(root, file))
                            if path_file.suffix == '.orig':
                                os.remove(os.path.join(root, path_file.stem))
                                os.rename(os.path.join(root, f'{path_file.stem}.orig'), os.path.join(root, path_file.stem))
                else:
                    print('Patching succeeded.')
