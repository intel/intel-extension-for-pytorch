import os
import platform
import glob
from ..utils._logger import logger, WarningType
import sys
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter
from tempfile import mkstemp
import uuid


def apply_monkey_patch(program, dtype, auto_ipex_verbose, disable_ipex_graph_mode):
    # Auto apply the ipex features
    # Open the original file and get the content
    with open(program) as f:
        original_program_lines = f.readlines()

    # Modify the content with import ipex
    monkey_patch = """import torch
import intel_extension_for_pytorch as ipex
from typing import Any, Callable
import functools
import threading

def set_optimized_attr(model):
    setattr(model, "_ipex_optimized", True)
    for child_name, child in model.named_children():
        set_optimized_attr(child)

_orig_module_call: Callable = torch.nn.Module.__call__

_auto_ipex_thread_local_storage = threading.local()
setattr(_auto_ipex_thread_local_storage, "nested_level", 0)

class nested_optimized(object):
    def __enter__(self):
        global _auto_ipex_thread_local_storage
        _auto_ipex_thread_local_storage.nested_level = getattr(_auto_ipex_thread_local_storage, 'nested_level', 0) + 1
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        global _auto_ipex_thread_local_storage
        _auto_ipex_thread_local_storage.nested_level -= 1
        return False

# _ipex_optimize_hit_count record how many times hit the optimized path.
# Only used when _auto_ipex_verbose is True.
_ipex_optimize_hit_count = 0

@functools.wraps(_orig_module_call)
def module_call_wrapper(mod, *args, **kwargs):
    def forward(mod, *args, **kwargs):
        # We skip the optimize path in below 2 cases to avoid performance overhead.
        # * Case 1: hasattr(mod, "_ipex_optimized"). For the module and submodules after
        # optimized, this attr will be added to avoid duplicated invocation.
        # * Case 2: nested_level != 0. Some modules in huggingface is created during
        # the forward function instead of the __init__ function. We are unable to add
        # the _ipex_optimized attr for this kind of module, so we will use the nested_level
        # to avoid unnecessary invocation for this kind of module.
        if not hasattr(mod, "_ipex_optimized") and (getattr(_auto_ipex_thread_local_storage, 'nested_level', 0)==0):

            set_optimized_attr(mod)
            dataType = torch.bfloat16 if ({0} == True) else torch.float32
            optimized_m = ipex.optimize(mod.eval(), dtype=dataType, graph_mode=(None if ({2} == True) else True)).eval()
            set_optimized_attr(optimized_m)

            def optimized_m_forward(*args, **kwargs):
                with torch.cpu.amp.autocast(enabled={0}), torch.no_grad(), nested_optimized():
                    return optimized_m(*args, **kwargs)

            if not {2}:
                # Warm up run to finish some warm up steps for graph mode in ipex.optimize
                for _ in range(3):
                    optimized_m_forward(*args, **kwargs)

            if {1}:
                # This path is valid only when auto_ipex_verbose is True.
                # And this path is only used for debug and UT.
                global _ipex_optimize_hit_count
                _ipex_optimize_hit_count += 1
                print("_ipex_optimize_hit_count is: %d" % _ipex_optimize_hit_count, flush=True)
                # Profile once to check whether ipex.optimize success or not
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                    optimized_m_forward(*args, **kwargs)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

            mod.forward = optimized_m_forward
        return _orig_module_call(mod, *args, **kwargs)
    return forward(mod, *args, **kwargs)

setattr(torch.nn.Module, "__call__", module_call_wrapper)\n""".format(
        dtype.lower() == "bfloat16", auto_ipex_verbose, disable_ipex_graph_mode
    )

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


def _exec(args):
    monkey_patch_program = apply_monkey_patch(
        args.program, args.dtype, args.auto_ipex_verbose, args.disable_ipex_graph_mode
    )
    try:
        cmd = []
        cmd.append(sys.executable)
        cmd.append("-u")
        cmd.append(monkey_patch_program)
        cmd.extend(args.program_args)
        cmd_s = " ".join(cmd)
        print("cmd_s is:{}".format(cmd_s))
        os.system(cmd_s)
    finally:
        # Remove the Monkey patched program
        if os.path.exists(monkey_patch_program):
            os.remove(monkey_patch_program)


def add_auto_ipex_params(parser, auto_ipex_default_enabled=False):
    group = parser.add_argument_group("Code_Free Parameters")
    group.add_argument(
        "--auto-ipex",
        "--auto_ipex",
        action="store_true",
        default=auto_ipex_default_enabled,
        help="Auto enabled the ipex optimization feature",
    )
    group.add_argument(
        "--dtype",
        metavar="\b",
        default="float32",
        type=str,
        choices=["float32", "bfloat16"],
        help="The data type to run inference. float32 or bfloat16 is allowed.",
    )
    group.add_argument(
        "--auto-ipex-verbose",
        "--auto_ipex_verbose",
        action="store_true",
        default=False,
        help="This flag is only used for debug and UT of auto ipex.",
    )
    group.add_argument(
        "--disable-ipex-graph-mode",
        "--disable_ipex_graph_mode",
        action="store_true",
        default=False,
        help="Enable the Graph Mode for ipex.optimize",
    )


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="This is a script for auto apply ipex optimization."
        "\n################################# Basic usage ############################# \n"
        "\n 1. Apply ipex optimization with fp32 data type\n"
        "\n   >>> python -m intel_extension_for_pytorch.cpu.auto_ipex python_script args \n"
        "\n 2. Apply ipex optimization with bf16 data type\n"
        "\n   >>> python -m intel_extension_for_pytorch.cpu.auto_ipex --dtype bfloat16 python_script args \n",
        formatter_class=RawTextHelpFormatter,
    )

    add_auto_ipex_params(parser, auto_ipex_default_enabled=True)

    # positional
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the proram/script to be launched. "
        "followed by all the arguments for the script",
    )
    # rest from the training program
    parser.add_argument("program_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    env_before = set(os.environ.keys())
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    args = parse_args()

    # Verify LD_PRELOAD
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            if item != "":
                matches = glob.glob(item)
                if len(matches) > 0:
                    lst_valid.append(item)
                else:
                    logger.warning(
                        f"You have set {item} into LD_PRELOAD but it doesn't exist. Removing it from LD_PRELOAD."
                        + "please install it if you want it or remove it from LD_PRELOAD if you don't",
                        _type=WarningType.MissingDependency,
                    )
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    _exec(args)

    for x in sorted(set(os.environ.keys()) - env_before):
        # Print the added ENV
        logger.debug("{0}={1}".format(x, os.environ[x]))


if __name__ == "__main__":
    main()
