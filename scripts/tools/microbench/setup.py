import os
import sys
import re
import glob
from subprocess import check_call
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension


torch_version = re.findall(r'[0-9]*\.[0-9]*\.[0-9]*', torch.__version__)[0]
print('torch_version: ', torch_version)
torch_version = torch_version.strip().replace('.', '_')
base_dir = os.path.dirname(os.path.abspath(__file__))
cmd = 'python codegen/{0}/generate.py '.format(torch_version)
cmd += '--declarations-path=codegen/{0}/Declarations.yaml '.format(
    torch_version)
cmd += '--out=csrc/generated.h'
check_call(cmd.split(' '), cwd=base_dir)


def get_defines(torch_version):
    table = {
        '1_7_0': ('TORCH_V_1_7', None),
        '1_11_0': ('TORCH_V_1_11', None),
        '1_10_0': ('TORCH_V_1_11', None),
    }
    return [table[torch_version]]


num_workers = int(os.environ['MAX_JOBS'])
cpp_parts = ['csrc/generated_' + str(t) + '.cpp' for t in range(num_workers)]

setup(
    name='microbench',
    ext_modules=[
        CppExtension(
            'microbench',
            ['csrc/microbench.cpp'] + cpp_parts,
            define_macros=get_defines(torch_version),
            extra_compile_args=['-Wno-unused-function']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
