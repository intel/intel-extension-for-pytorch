import os
import shutil
import subprocess
import sys


def check_flake8_errors(base_dir, filepath):
    if shutil.which('flake8') is None:
        print("WARNING: Please install flake8 by pip install -r requirements-flake8.txt to check format!")
    flak8_cmd = ['flake8']  # '--quiet'
    if os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if (file.endswith('.py')):
                    flak8_cmd.append(os.path.join(root, file))
    elif os.path.isfile(filepath):
        flak8_cmd.append(filepath)
    ret = subprocess.call(flak8_cmd, cwd=base_dir)
    return ret

if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    setupfile = os.path.join(base_dir, 'setup.py')
    base_pydir = os.path.join(base_dir, 'intel_extension_for_pytorch')
    base_scripts = os.path.join(base_dir, 'scripts')
    base_examples = os.path.join(base_dir, 'tests/gpu/examples')

    Check_dir = [setupfile, base_pydir, base_examples]
    ret = sum([check_flake8_errors(base_dir, path) for path in Check_dir])
    if ret != 0:
        print("ERROR: flake8 found format errors!")
        sys.exit(1)
    else:
        print("Pass!")
