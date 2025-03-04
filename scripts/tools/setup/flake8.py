import os
import shutil
import subprocess
import sys

_IS_WINDOWS = sys.platform == "win32"


def check_flake8_errors(base_dir, filepath):
    if shutil.which("flake8") is None:
        return -1
    flak8_cmd = ["flake8"]  # '--quiet'

    if shutil.which("black") is None:
        return -1
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

    ret_blk = 0
    # Auto format python code.
    try:
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
    except subprocess.CalledProcessError as e:
        print(e.output)
        return 1


if __name__ == "__main__":
    if _IS_WINDOWS:
        print("skip flake8 check for Windows")
        sys.exit(0)

    base_dir = os.path.abspath(
        os.path.dirname(os.path.join(os.path.abspath(__file__), "../../../../"))
    )
    setupfile = os.path.join(base_dir, "setup.py")
    base_pydir = os.path.join(base_dir, "intel_extension_for_pytorch")
    base_scripts = os.path.join(base_dir, "scripts")
    base_cpu_uts = os.path.join(base_dir, "tests/cpu")
    base_cpu_example = os.path.join(base_dir, "examples/cpu")

    Check_dir = [setupfile, base_pydir, base_scripts, base_cpu_uts, base_cpu_example]
    ret = sum([check_flake8_errors(base_dir, path) for path in Check_dir])
    if ret > 0:
        print("ERROR: flake8 found format errors!")
        sys.exit(1)
    elif ret < 0:
        print(
            "WARNING: Please install flake8 by pip install -r requirements-flake8.txt to check format!"
        )
    else:
        print("Pass!")
