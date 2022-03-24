import subprocess
import platform
import sys
import os

def check_minimal_isa_support():
    def get_cpu_info():
        cpu_info_path = "/proc/cpuinfo"
        if os.path.exists(cpu_info_path):
            cpu_info_command = "cat {}".format(cpu_info_path)
            all_sub_cpu_info = subprocess.getoutput(cpu_info_command).strip()
            for sub_cpu_info in all_sub_cpu_info.split("\n"):
                if sub_cpu_info.startswith("flags"):
                    cpu_flags = sub_cpu_info.replace("\t", '').upper().split(":")
                    assert len(cpu_flags) >= 2
                    all_cpu_flags = cpu_flags[1].split(" ")
                    return all_cpu_flags

            return []
        else:
            sys.exit("The extension does not support current platform - {}.".format(platform.system()))

    def check_avx2_support():
        cpu_flags = get_cpu_info()
        minimal_binary_isa = "AVX2"
        if minimal_binary_isa not in cpu_flags:
            return False

        return True

    err_msg = "ERROR! Intel® Extension for PyTorch* only works on machines with instruction sets equal or newer than AVX2, which are not detected on the current machine."
    if not check_avx2_support():
        sys.exit(err_msg)