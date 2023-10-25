import os
import shutil
import subprocess
import sys


def check_flake8_errors(base_dir, filepath):
    if shutil.which("flake8") is None:
        print(
            "WARNING: Please install flake8 by pip install -r requirements-flake8.txt to check format!"
        )
        return 1
    flak8_cmd = ["flake8"]  # '--quiet'

    if shutil.which("black") is None:
        print(
            "WARNING: Please install black by pip install -r requirements-flake8.txt to auto format!"
        )
        return 1
    black_cmd = ["black"]

    if os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.endswith(".py"):
                    black_cmd.append(os.path.join(root, file))
                    flak8_cmd.append(os.path.join(root, file))
    elif os.path.isfile(filepath):
        black_cmd.append(filepath)
        flak8_cmd.append(filepath)

    # Auto format python code.
    blk_output = subprocess.check_output(
        black_cmd,
        cwd=base_dir,
        stderr=subprocess.STDOUT,
    )
    output_string = blk_output.decode("utf-8")
    print(output_string)
    if output_string.find("reformatted") == -1:
        ret_blk = 0
    else:
        ret_blk = 1

    # Check code style.
    ret_flak8 = subprocess.call(flak8_cmd, cwd=base_dir)
    status_code = ret_flak8 + ret_blk
    print("status code: ", status_code)

    return status_code


if __name__ == "__main__":
    base_dir = os.path.abspath(
        os.path.dirname(os.path.join(os.path.abspath(__file__), "../../../../"))
    )
    setupfile = os.path.join(base_dir, "setup.py")
    base_pydir = os.path.join(base_dir, "intel_extension_for_pytorch")
    base_scripts = os.path.join(base_dir, "scripts")
    base_examples = os.path.join(base_dir, "tests/gpu/examples")
    base_native = os.path.join(base_dir, "tests/gpu/native")
    base_regression = os.path.join(base_dir, "tests/gpu/regression")

    Check_dir = [setupfile, base_pydir, base_examples, base_native, base_regression]
    ret = sum([check_flake8_errors(base_dir, path) for path in Check_dir])
    if ret != 0:
        print("ERROR: flake8 found format errors!")
        sys.exit(1)
    else:
        print("Pass!")
