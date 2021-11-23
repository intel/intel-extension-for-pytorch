import subprocess
import platform
import sys
import os

def _check_avx_isa(binary_isa):
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

    def check_isa(binary_isa):
        cpu_flags = get_cpu_info()
        binary_isa = binary_isa.upper()
        if binary_isa == "AVX2":
            if binary_isa not in cpu_flags:
                return False
        else:
            avx512_isa = ["avx512f", "avx512bw", "avx512vl", "avx512dq"]
            for avx512_sub_isa in avx512_isa:
                if avx512_sub_isa.upper() not in cpu_flags:
                    return False
        return True

    err_msg = "The extension binary is {} while current machine does not support {}."
    if not check_isa(binary_isa):
        sys.exit(err_msg.format(binary_isa, binary_isa))
