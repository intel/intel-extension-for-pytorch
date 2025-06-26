#!/usr/bin/env python
# encoding: utf-8

# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html

import argparse
import os
from pathlib import Path
import platform
import re
import sys
import time
import urllib.parse

UTILSFILENAME = "compilation_utils.py"
SYSTEM = platform.system()
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = SCRIPTDIR
SRCDIR = ""
if (
    Path(BASEDIR).parts[-1] == "scripts"
    and os.path.isdir(os.path.join(BASEDIR, "..", "intel_extension_for_pytorch"))
    and os.path.exists(os.path.join(BASEDIR, "..", "setup.py"))
):
    dir_parts = Path(BASEDIR).parts
    SRCDIR = dir_parts[-2]
    BASEDIR = os.path.join(*dir_parts[:-2])
if SRCDIR == "":
    SRCDIR = "intel-extension-for-pytorch"


def _get_whl_from_dist(directory):
    whl_files = []
    for f in os.listdir(directory):
        if Path(f).suffix == ".whl":
            whl_files.append(f)
    assert (
        len(whl_files) == 1
    ), f"{len(whl_files)} files are found in {directory}, expect 1."
    return os.path.join(directory, whl_files[0])


def _compile_base(cmd, cwd, env, show_command):
    if show_command:
        print(env)
    redirect_file = os.path.join(cwd, "build.log")
    redirect_append = False
    if env is not None:
        if os.path.exists(redirect_file):
            os.remove(redirect_file)
        with open(redirect_file, "w") as file:
            file.write(
                "******************** Environment Variables ********************\n"
            )
            for key, value in env.items():
                file.write(f"{key}: {value}\n")
            file.write(
                "***************************************************************\n"
            )
        redirect_append = True
    exec_cmds(
        cmd,
        cwd=cwd,
        env=env,
        redirect_file=redirect_file,
        redirect_append=redirect_append,
        show_command=show_command,
    )


def _compile(directory, env, pkg_name="", incremental=False, show_command=False):
    print(f"========== {directory} ==========")
    dir_base = os.path.join(BASEDIR, directory)
    if not incremental:
        exec_cmds(
            "python setup.py clean", cwd=dir_base, env=env, show_command=show_command
        )
    dir_dist = os.path.join(dir_base, "dist")
    remove_file_dir(dir_dist)
    _compile_base(
        "python setup.py bdist_wheel", cwd=dir_base, env=env, show_command=show_command
    )
    exec_cmds(
        f"python -m pip install {_get_whl_from_dist(dir_dist)}",
        dir_base,
        env=env,
        show_command=show_command,
    )


def process(*args):
    args_max_jobs = args[0]
    args_ver_ipex = args[1]
    args_with_vision = args[2]
    args_with_audio = args[3]
    args_incremental = args[4]
    args_verbose = args[5]

    global exec_cmds
    global check_system_commands
    global remove_file_dir
    global clear_directory
    global update_source_code
    global source_env
    global get_duration
    global download
    utils_filepath = (
        os.path.join(BASEDIR, SRCDIR, "tools", UTILSFILENAME)
        if BASEDIR != SCRIPTDIR
        else os.path.join(BASEDIR, UTILSFILENAME)
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("script_module", utils_filepath)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules["script_module"] = utils_module
    spec.loader.exec_module(utils_module)
    exec_cmds = utils_module.exec_cmds
    check_system_commands = utils_module.check_system_commands
    remove_file_dir = utils_module.remove_file_dir
    clear_directory = utils_module.clear_directory
    update_source_code = utils_module.update_source_code
    source_env = utils_module.source_env
    get_duration = utils_module.get_duration
    download = utils_module.download

    n_cores = os.cpu_count()
    if args_max_jobs == 0:
        args_max_jobs = n_cores
    elif args_max_jobs > n_cores:
        print(
            "--max-jobs argument is set a value larger than number of available cores. Reset it to the number of available cores."
        )
        args_max_jobs = n_cores
    else:
        pass

    # Check existence of required system commands
    commands = ["git", "gcc", "g++"]
    check_system_commands(commands)

    durations = {}

    # Update IPEX source code
    t0 = int(time.time() * 1000)
    update_source_code(
        SRCDIR,
        "https://github.com/intel/intel-extension-for-pytorch.git",
        args_ver_ipex,
        basedir=BASEDIR,
        show_command=args_verbose,
    )
    durations["Retrieve IPEX source code"] = get_duration(t0)

    # Retrieve dependency information
    sys.path.append(os.path.join(BASEDIR, SRCDIR, "tools"))
    from dep_ver_utils import process_file as dep_info_retrieve

    INFO_GCC = dep_info_retrieve(
        os.path.join(BASEDIR, SRCDIR, "dependency_version.json"), "gcc"
    )
    INFO_TORCH = dep_info_retrieve(
        os.path.join(BASEDIR, SRCDIR, "dependency_version.json"), "pytorch"
    )
    INFO_TORCHVISION = dep_info_retrieve(
        os.path.join(BASEDIR, SRCDIR, "dependency_version.json"), "torchvision:version"
    )
    INFO_TORCHAUDIO = dep_info_retrieve(
        os.path.join(BASEDIR, SRCDIR, "dependency_version.json"), "torchaudio:version"
    )
    if args_verbose:
        print(f"INFO_TORCH:       {str(INFO_TORCH)}")
        print(f"INFO_TORCHVISION: {str(INFO_TORCHVISION)}")
        print(f"INFO_TORCHAUDIO:  {str(INFO_TORCHAUDIO)}")

    if INFO_TORCHVISION == "N/A":
        args_with_vision = False
    if INFO_TORCHAUDIO == "N/A":
        args_with_audio = False

    exec_cmds("python -m pip install packaging", silent=True, shell=True)
    from packaging import version

    _, line_stdout = exec_cmds("gcc -dumpfullversion", silent=True, shell=True)
    assert len(line_stdout) == 1, f"Unexpected gcc version: {line_stdout}"
    VER_GCC = line_stdout[0]
    if version.parse(VER_GCC) < version.parse(INFO_GCC["min-version"]):
        print(
            "Warning: Current gcc version ({VER_GCC}) is older than the expected minimum version "
            + f'({INFO_GCC["min-version"]}).'
        )
        time.sleep(5)
    del sys.modules["packaging"]
    exec_cmds("python -m pip uninstall -y packaging", silent=True, shell=True)

    # Clean Python environment
    t0 = int(time.time() * 1000)
    exec_cmds(
        "python -m pip uninstall -y torch torchvision torchaudio",
        show_command=args_verbose,
    )
    exec_cmds(
        """python -m pip uninstall -y intel-extension-for-pytorch
                 python -m pip install cmake make ninja requests""",
        shell=True,
        show_command=args_verbose,
    )
    durations["Clean Python environment"] = get_duration(t0)

    # Prepare compilation environment
    t0 = int(time.time() * 1000)
    env = os.environ.copy()
    env["MAX_JOBS"] = str(args_max_jobs)
    durations["Prepare compilation environment"] = get_duration(t0)

    # Install PyTorch/TorchVision/TorchAudio
    t0 = int(time.time() * 1000)
    INDEX_URL = INFO_TORCH["index-url"]
    if "nightly" in INDEX_URL:
        if args_with_vision:
            if INFO_TORCHVISION != "":
                INFO_TORCHVISION = f"=={INFO_TORCHVISION}"
            exec_cmds(
                f"python -m pip install torchvision{INFO_TORCHVISION} --index-url {INDEX_URL}",
                show_command=args_verbose,
            )
        if args_with_audio:
            if INFO_TORCHAUDIO != "":
                INFO_TORCHAUDIO = f"=={INFO_TORCHAUDIO}"
            exec_cmds(
                f"python -m pip install torchaudio{INFO_TORCHAUDIO} --index-url {INDEX_URL}",
                show_command=args_verbose,
            )
        exec_cmds(
            f'python -m pip install torch=={INFO_TORCH["version"]} --index-url {INDEX_URL}',
            show_command=args_verbose,
        )
    else:
        command = f'python -m pip install torch=={INFO_TORCH["version"]}'
        if args_with_vision:
            if INFO_TORCHVISION != "":
                INFO_TORCHVISION = f"=={INFO_TORCHVISION}"
            command += f" torchvision{INFO_TORCHVISION}"
        if args_with_audio:
            if INFO_TORCHAUDIO != "":
                INFO_TORCHAUDIO = f"=={INFO_TORCHAUDIO}"
            command += f" torchaudio{INFO_TORCHAUDIO}"
        command += f" --index-url {INDEX_URL}"
        exec_cmds(command, show_command=args_verbose)
    durations["Install PyTorch packages"] = get_duration(t0)

    # Install Intel® Extension for PyTorch*
    t0 = int(time.time() * 1000)
    env_ipex = env.copy()
    exec_cmds(
        "python -m pip install -r requirements.txt",
        cwd=os.path.join(BASEDIR, SRCDIR),
        show_command=args_verbose,
    )
    env_ipex["BUILD_WITH_CPU"] = "1"
    _compile(
        SRCDIR,
        env_ipex,
        pkg_name="intel_extension_for_pytorch",
        incremental=args_incremental,
        show_command=args_verbose,
    )
    durations["Compile IPEX"] = get_duration(t0)

    # Print step duration
    print("")
    print("******************** Compilation Finished ********************")
    for key in sorted(durations):
        print(f"{key}: {durations[key]:.2f}s")
    print("")

    # Sanity Test
    print("************************* Sanity Test ************************")
    if SYSTEM == "Linux":
        _, libstdcpp = exec_cmds(
            f"bash ./{SRCDIR}/tools/get_libstdcpp_lib.sh",
            cwd=BASEDIR,
            silent=True,
            show_command=args_verbose,
        )
        assert len(libstdcpp) == 1, "Something goes wrong when finding libstdcpp"
        if not libstdcpp[0].startswith("/usr/lib/"):
            os.environ["LD_PRELOAD"] = libstdcpp[0]
            print(
                f'Note: Set environment variable "export LD_PRELOAD={libstdcpp[0]}"'
                + ' to avoid the "version `GLIBCXX_N.N.NN\' not found" error.'
            )
            print("")

    import torch

    print(f"torch_version:       {torch.__version__}")
    print(f"torch_cxx11_abi:     {str(int(torch._C._GLIBCXX_USE_CXX11_ABI))}")
    if args_with_vision:
        import torchvision

        print(f"torchvision_version: {torchvision.__version__}")
    if args_with_audio:
        import torchaudio

        print(f"torchaudio_version:  {torchaudio.__version__}")
    import intel_extension_for_pytorch as ipex

    print(f"ipex_version:        {ipex.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to simply setting up Intel® Extension for PyTorch* environment, "
        + "including installation/compilation of PyTorch and/or TorchVision/TorchAudio."
    )
    parser.add_argument(
        "--max-jobs",
        help="Number of cores used for the compilation. Setting it to 0 for automatically detection. Value is 0 by default.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ver-ipex",
        help="Designate a specific branch/tag of Intel® Extension for PyTorch* source code to build with.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--with-vision",
        help="Install TorchVision.",
        action="store_true",
    )
    parser.add_argument(
        "--with-audio",
        help="Install TorchAudio.",
        action="store_true",
    )
    parser.add_argument(
        "--incremental",
        help="Enable IPEX incremental compilation.",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="Show more information for debugging compilation.",
        action="store_true",
    )
    args = parser.parse_args()

    utils_filepath = os.path.join(BASEDIR, UTILSFILENAME)
    if BASEDIR != SCRIPTDIR:
        assert (
            args.ver_ipex == ""
        ), "Argument --ver-ipex cannot be set if you run the script from a exisiting source code directory."
    else:
        assert (
            args.ver_ipex != ""
        ), "Argument --ver-ipex must be set to a branch/tag/commit of Intel® Extension for PyTorch* source code."
        if os.path.isfile(utils_filepath):
            os.remove(utils_filepath)
        url = f"https://github.com/intel/intel-extension-for-pytorch/blob/{urllib.parse.quote(args.ver_ipex)}/tools/{UTILSFILENAME}"
        import subprocess

        p = subprocess.Popen(
            "python -m pip install requests",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )
        del sys.modules["subprocess"]
        for line in iter(p.stdout.readline, ""):
            pass
        import requests

        response = requests.get(url)
        assert (
            response.status_code >= 200 and response.status_code < 300
        ), f"Failed to access {url}, status code: {response.status_code}."
        urls = re.findall('"rawBlobUrl":"(.*?)"', response.text)
        assert len(urls) == 1, f"Unexpected number of raw URLs retrieved.\n{matches}"
        response = requests.get(urls[0], stream=True)
        with open(utils_filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        del sys.modules["requests"]

    process(
        args.max_jobs,
        args.ver_ipex,
        args.with_vision,
        args.with_audio,
        args.incremental,
        args.verbose,
    )

    for item in [os.path.join(BASEDIR, "__pycache__"), utils_filepath]:
        remove_file_dir(item)
