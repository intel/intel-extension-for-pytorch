"""Helper to disassemble the GPU code"""
import os
import re
from subprocess import check_call
import json
import argparse


def read_compile_commands(file):
    with open(file) as json_file:
        compile_database = json.load(json_file)

    print("compiled objects number ", len(compile_database))
    return compile_database


def get_compile_options(command):
    options = re.match(r'^(\S*)(\s*)(.*)(-o)', command)
    return options.groups()[2]


def disassemble(file):
    pass


def get_tmp_file(name, suffix):
    return "/tmp/{}-{}.{}".format(name, os.getpid(), suffix)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Disassemble the ipex kernel code')
    parser.add_argument('--compile-commands-json',
                        help='compile_commands.json path for ipex',
                        default="./build/compile_commands.json",
                        type=str)
    parser.add_argument('--out-dir',
                        help='path to output directory',
                        default="./disassemble",
                        type=str)
    parser.add_argument('--source',
                        help='source to disassemble',
                        default="None",
                        type=str)
    parser.add_argument('--obj',
                        help='object to disassemble',
                        default=None,
                        type=str)
    parser.add_argument('--device',
                        help='object to disassemble',
                        default="pvc",
                        type=str)
    args = parser.parse_args()

    ipex_base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../../'
    cwd = os.getcwd()
    args.out_dirs = os.path.abspath(args.out_dir)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    def run(commands, env=None, cwd=cwd):
        "Executes cmake with arguments and an environment."

        print(' '.join(commands))
        # check_call(command, cwd=self.build_dir, env=env)
        my_env = os.environ.copy()
        if env is not None:
            for k in env:
                my_env[k] = env[k]

        check_call(commands, cwd=cwd, env=my_env)

    if args.obj is None:
        compile_database = read_compile_commands(args.compile_commands_json)
    else:
        print("disassemble of ", args.obj)
        print("director of ", os.path.split(args.obj))
        paths = os.path.split(args.obj)
        basename = os.path.splitext(paths[-1])[0]
        print("basename of ", basename)

        gpu_llvm = get_tmp_file(basename, "bc")
        host_obj = get_tmp_file(basename, "o")

        # unbundle the object
        run(['clang-offload-bundler',
             '-type=o',
             '-targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown-sycldevice',
             '-inputs={}'.format(args.obj),
             '-outputs={},{}'.format(host_obj, gpu_llvm),
             '-unbundle',
             '-allow-missing-bundles'])

        # sycl post link
        gpu_post_link_llvm = get_tmp_file(basename, "post.bc")
        run(['sycl-post-link',
             '-split=auto',
             '-lower-esimd',
             '-spec-const=default',
             '--ir-output-only',
             '-o',
             '{}'.format(gpu_post_link_llvm),
             '{}'.format(gpu_llvm)])

        # convert llvm to SPIRV
        gpu_spv = get_tmp_file(basename, "spv")
        run(['llvm-spirv',
             '-spirv-ext=+all',
             '{}'.format(gpu_post_link_llvm),
             '-o={}'.format(gpu_spv)])

        # use ocloc to disassemble the GPU code
        gpu_bin = get_tmp_file(basename, "spv")
        run(['ocloc',
             '-file',
             '{}'.format(gpu_spv),
             '-spirv_input',
             '-device',
             '{}'.format(args.device)],
            env={'IGC_ShaderDumpEnable': '1', 'IGC_DumpToCurrentDir': '1'},
            cwd=args.out_dirs)
