import platform
import subprocess
import os
import sys
import logging
from tempfile import mkstemp
import uuid
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter


format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)


def apply_monkey_patch(args):
    # Auto apply the ipex features
    # Open the original file and get the content
    program = args.program
    with open(program) as f:
        original_program_lines = f.readlines()

    # Modify the content with import ipex
    monkey_patch = """import torch
import intel_extension_for_pytorch as ipex
"""
    if args.convert_fp64_to_fp32:
        monkey_patch += """ipex.xpu.overrides.convert_default_dtype(torch.float64, torch.float32, True)
"""
    original_program_lines.insert(0, monkey_patch)

    program_absolute_path = os.path.abspath(program)
    program_absolute_path_dir = os.path.dirname(program_absolute_path)
    generate_file_suffix = (
        str(hash(program_absolute_path)) + str(uuid.uuid1()) + "_auto_ipex"
    )
    _, generate_file = mkstemp(
        suffix=generate_file_suffix, dir=program_absolute_path_dir, text=True
    )

    # Write the monkey_patched content to temp file
    with open(generate_file, "w") as f:
        f.writelines(original_program_lines)

    return generate_file


class Launcher:
    r"""
    Base class for launcher
    """

    def __init__(self):
        pass

    def launch(self, args):
        pass

    def logger_env(self, env_name=""):
        if env_name in os.environ:
            logger.info("{}={}".format(env_name, os.environ[env_name]))

    def set_env(self, env_name, env_value=None):
        if not env_value:
            logger.warning("{} is None".format(env_name))
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning(
                "{} in environment variable is {} while the value you set is {}".format(
                    env_name, os.environ[env_name], env_value
                )
            )
        self.logger_env(env_name)


class XPUDefaultLauncher(Launcher):
    """
    Run the program using XPU.
    # Note: For now, we only support single instance in this script
    """

    def launch(self, args):
        processes = []
        cmd = []

        monkey_program = apply_monkey_patch(args)

        cmd.append(sys.executable)
        cmd.append(monkey_program)
        cmd.extend(args.program_args)

        cmd_s = " ".join(cmd)
        process = subprocess.Popen(cmd_s, env=os.environ, shell=True)
        processes.append(process)
        try:
            for process in processes:
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode=process.returncode, cmd=cmd_s
                    )
        except subprocess.CalledProcessError as e:
            print(e.output)
        finally:
            os.remove(monkey_program)


def init_parser(parser):
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """

    # positional
    parser.add_argument(
        "--convert-fp64-to-fp32",
        "--convert_fp64_to_fp32",
        action="store_true",
        dest="convert_fp64_to_fp32",
        help="To automatically convert torch.float64(double) dtype to torch.float32",
    )
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the proram/script to be launched. "
        "followed by all the arguments for the script",
    )

    # rest from the training program
    parser.add_argument("program_args", nargs=REMAINDER)
    return parser


def run_main_with_args(args):
    env_before = set(os.environ.keys())
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")
    launcher = None
    launcher = XPUDefaultLauncher()
    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug("{0}={1}".format(x, os.environ[x]))


def main():
    parser = ArgumentParser(
        description="This is a script for launching PyTorch training and inference on Intel GPU Series"
        "with optimal configurations. "
        "\n################################# Basic usage ############################# \n"
        "\n 1. Run with args\n"
        "\n   >>> ipexrun xpu python_script args \n"
        "\n############################################################################# \n",
        formatter_class=RawTextHelpFormatter,
    )
    parser = init_parser(parser)
    args = parser.parse_args()
    run_main_with_args(args)


if __name__ == "__main__":
    main()
