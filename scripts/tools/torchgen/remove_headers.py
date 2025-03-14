import argparse
import re

parser = argparse.ArgumentParser(description="Utils for remove unused headers")
parser.add_argument("--register_xpu_path", type=str, help="file location of RegisterXPU.cpp")
args = parser.parse_args()

def replace_op_headers():
    with open(args.register_xpu_path, 'r') as fr:
        lines = fr.readlines()
        patt = r'#include <ATen/ops'
        rep = r'#include <xpu/ATen/ops'
        with open(args.register_xpu_path, 'w') as fw:
            for ln in lines:
                if 'empty.h' in ln or 'empty_strided.h' in ln or 'as_strided_native.h' in ln:
                    continue
                if 'copy_from_and_resize.h' in ln or 'copy_from.h' in ln:
                    continue
                replaced = re.sub(patt, rep, ln)
                fw.write(replaced)

if __name__ == "__main__":
    replace_op_headers()
