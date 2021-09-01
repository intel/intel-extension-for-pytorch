import os
import sys
import subprocess
import argparse


def file_exists(filename):
    if os.path.exists(filename):
        return True
    else:
        return False


def compare(file1, file2):
    fp1 = open(file1, 'r')
    fp2 = open(file2, 'r')
    file_list1 = [i for i in fp1]
    file_list2 = [i for i in fp2]
    if (len(file_list1) != len(file_list2)):
        return False
    else:
        for line in zip(file_list1, file_list2):
            if line[0] != line[1]:
                return False
        return True


def gpu_gen_and_should_copy(install_dir):
    cwd = os.path.dirname(os.path.abspath(__file__))
    generate_code_cmd = ['python',
                         'gen-gpu-decl.py',
                         '--gpu_decl=./',
                         'DPCPPGPUType.h',
                         'QUANTIZEDDPCPPGPUType.h',
                         'DedicateType.h',
                         'DispatchStubOverride.h',
                         'RegistrationDeclarations.h']
    if subprocess.call(generate_code_cmd, cwd=cwd) != 0:
        print("Failed to run '{}'".format(generate_code_cmd), file=sys.stderr)
        sys.exit(1)

    generate_code_cmd = ['python',
                         'gen-gpu-ops.py',
                         '--output_folder=./',
                         'DPCPPGPUType.h',
                         'QUANTIZEDDPCPPGPUType.h',
                         'RegistrationDeclarations_DPCPP.h',
                         'Functions_DPCPP.h']
    if subprocess.call(generate_code_cmd, cwd=cwd) != 0:
        print("Failed to run '{}'".format(generate_code_cmd), file=sys.stderr)
        sys.exit(1)

    files_name = ['aten_ipex_type_default.cpp.in', 'aten_ipex_type_default.h.in', 'aten_ipex_type_dpcpp.h.in']
    generated_files = [os.path.join(os.path.abspath(cwd), file) for file in files_name]
    source_files = [os.path.join(os.path.abspath(install_dir), os.path.splitext(file)[0]) for file in files_name]

    for gen_file, source_file in zip(generated_files, source_files):
        # if file_exists(source_file):
        #     if compare(gen_file, source_file):
        #         continue
        subprocess.call(['cp', gen_file, source_file], cwd=cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ATen source files')
    parser.add_argument(
        '-d', '--install_dir', help='output directory', default='.')

    options = parser.parse_args()

    gpu_gen_and_should_copy(options.install_dir)
