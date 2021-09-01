"Helper to disassemble the GPU code"
import multiprocessing
import os
import re
from subprocess import check_call, check_output
import sys
import distutils
import distutils.sysconfig
from distutils.version import LooseVersion
import json


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
