#!/usr/bin/env python
# encoding: utf-8

import os
import platform
import shutil
import subprocess
import stat
import sys
import time

SYSTEM = platform.system()

def _exec_cmd_single(command,
                     cwd = None,
                     env = None,
                     shell = False,
                     level = 1,
                     exit_on_failure = True,
                     show_command = False,
                     silent = False,
                     redirect_file = '',
                     redirect_append = False):
    if command == '':
        return 0, []
    file = None
    if redirect_file != '':
        if redirect_append:
            file = open(redirect_file, 'a')
        else:
            file = open(redirect_file, 'w')
    cmd = None
    if shell:
        cmd = command.strip()
    else:
        cmd = command.strip().split()
    if show_command:
        if not silent:
            print(f'{"+" * level} {command}')
        if not file is None:
            file.write(f'{"+" * level} {command}\n')
    lines_stdout = []
    p = subprocess.Popen(cmd,
                         cwd = cwd,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.STDOUT,
                         env = env,
                         shell = shell,
                         text = True)
    for line in iter(p.stdout.readline, ''):
        lines_stdout.append(line.strip())
        if not silent:
            print(line, end = '')
        if not file is None:
            file.write(line)
    p.stdout.close()
    return_code = p.wait()
    if file is not None:
        file.close()
    if exit_on_failure:
        assert p.returncode == 0, f'Command [{command.strip()}] execution failed.'
    return p.returncode, lines_stdout

def exec_cmds(commands,
              cwd = None,
              env = None,
              shell = False,
              exit_on_failure = True,
              stop_on_failure = True,
              show_command = False,
              silent = False,
              redirect_file = '',
              redirect_append = False):
    if show_command:
        silent = False
    if redirect_file != '':
        if not redirect_append and os.path.exists(redirect_file):
            os.remove(redirect_file)
        redirect_append = True
    ret_code = 0
    ret_lines = []
    cmds = []
    for c in commands.split('\n'):
        c = c.strip()
        if c != '':
            cmds.append(c)
    for cmd in cmds:
        r, lines_stdout = _exec_cmd_single(cmd,
                                           cwd = cwd,
                                           env = env,
                                           shell = shell,
                                           show_command = show_command,
                                           exit_on_failure = exit_on_failure,
                                           silent = silent,
                                           redirect_file = redirect_file,
                                           redirect_append = redirect_append)
        ret_code = r
        ret_lines += lines_stdout
        if stop_on_failure and r > 0 and len(cmds) > 1:
            print('Execution failed!')
            break
    return ret_code, ret_lines

def check_system_commands(commands):
    absent = 0
    ret = {}
    for cmd in commands:
        ret[cmd] = shutil.which(cmd)
        if not ret[cmd]:
            print(f'{cmd} not found.')
            absent += 1
    assert absent == 0, 'Required system command(s) not found.'
    return ret

def remove_directory(directory):
    if SYSTEM == 'Windows':
        for root, dirs, files in os.walk(directory):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.chmod(dir_path, stat.S_IWRITE)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.chmod(file_path, stat.S_IWRITE)
    shutil.rmtree(directory)

def remove_file_dir(item):
    if os.path.isfile(item):
        os.remove(item)
    elif os.path.isdir(item):
        remove_directory(item)
    else:
        pass

def clear_directory(directory):
    for item in os.listdir(directory):
        if item.startswith("."):
            continue
        remove_file_dir(os.path.join(directory, item))

def update_source_code(dir_name,
                       url_repo,
                       branch,
                       branch_main = 'main',
                       basedir = '',
                       show_command = False):
    print(f'========== {dir_name} ==========')
    dir_target = os.path.join(basedir, dir_name)
    if not os.path.isdir(dir_target):
        exec_cmds(f'git clone {url_repo} {dir_name}',
                  cwd=basedir,
                  show_command = show_command)
    if branch != '':
        clear_directory(dir_target)
        exec_cmds(f'''git checkout .
                      git checkout {branch_main}
                      git pull''',
                  cwd = dir_target,
                  silent = True)
        exec_cmds(f'''git checkout {branch}
                      git pull''',
                  cwd = dir_target,
                  show_command = show_command)
    exec_cmds('''git submodule sync
                 git submodule update --init --recursive''',
              cwd = dir_target,
              show_command = show_command)

def source_env(script,
                env = None,
                show_command = False):
    if script == '' or not os.path.exists(script):
        print(f'Incorrect script: {script}')
        return None
    if env is None:
        env = os.environ.copy()
    separator = '========== SEPARATOR =========='
    command = ''
    if SYSTEM == 'Linux':
        command = f'. {script} && echo "{separator}" && env'
    elif SYSTEM == 'Windows':
        command = f'cmd.exe /c ""{script}" && echo {separator} && set"'
    else:
        pass
    if show_command:
        print(f'+ {command.split("&&")[0].strip()}')
    _, lines_stdout = exec_cmds(command,
                                 env = env,
                                 shell = True,
                                 silent = True)
    parse_start = False
    for line in lines_stdout:
        if parse_start:
            key, value = line.strip().split('=', 1)
            env[key] = value
        if line.strip() == separator:
            parse_start = True
    return env

def get_duration(t0):
    t1 = int(time.time() * 1000)
    return (t1 - t0) / 1000.0

def download(url, filepath):
    import requests
    response = requests.get(url, stream=True)
    assert response.status_code >= 200 and response.status_code < 300, f'Failed to download {url}.'
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
