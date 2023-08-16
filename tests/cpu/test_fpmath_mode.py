import unittest
import os
import subprocess
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex
import torch
import itertools
from functools import wraps

def fpmath_mode_env(func):
    @wraps(func)
    def wrapTheFunction(*args):
        func(*args)
        # set the fp32_math_mode back to FP32 after each UT
        ipex.set_fp32_math_mode()
    return wrapTheFunction

class TestFPMathCases(TestCase):
    @fpmath_mode_env
    def test_set_and_get_fpmath(self):
        fpmath_mode = [ipex.FP32MathMode.BF32, ipex.FP32MathMode.FP32]
        for mode in fpmath_mode:
            ipex.set_fp32_math_mode(mode=mode, device="cpu")
            assert ipex.get_fp32_math_mode() == mode, \
            "The return value of get_fpmath_mode is different from the value passed to set_fpmath_mode."
        ipex.set_fp32_math_mode()
        assert ipex.get_fp32_math_mode() == ipex.FP32MathMode.FP32, \
            "The default fp32_math_mode should be LowFP32MathMode.FP32."

    @fpmath_mode_env
    def test_fpmath_bf32(self):
        modes = ["jit", "imperative"]
        bias = [True, False]
        for mode, b in itertools.product(modes, bias):
            num1, num2, num3, num4, num5, num6, num7 = 0, 0, 0, 0, 0, 0, 0
            loc = os.path.dirname(os.path.abspath(__file__))
            cmd = 'DNNL_VERBOSE=1 python -u {}/fpmath_mode.py --mode="{}" --fpmath="BF32" --bias={}'.format(loc, mode, b)
            with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
                for line in p.stdout.readlines():
                    line = str(line, 'utf-8').strip()
                    if "attr-fpmath:bf16" in line and "convolution" in line:
                        num1 = num1 + 1
                        if "backward" in line:
                            num4 = num4 + 1
                    if "attr-fpmath:bf16" in line and "inner_product" in line:
                        num2 = num2 + 1
                        if "backward" in line:
                            num5 = num5 + 1
                    if "attr-fpmath:bf16" in line and "matmul" in line:
                        num3 = num3 + 1
                    if "attr-fpmath:bf16" in line and "rnn" in line:
                        num6 = num6 + 1
                        if "backward" in line:
                            num7 = num7 + 1
            assert num1 > 0 and num2 > 0 and num3 > 0 and num6 > 0, 'The implicit FP32 to BF16 data type conversion failed to enable.'
            assert num4 > 0 and num5 > 0 and num3 >= 3 and num7 > 0, 'The implicit FP32 to BF16 data type conversion failed to enable in backward pass.'

    @fpmath_mode_env
    def test_fpmath_strict(self):
        modes = ["jit", "imperative"]
        bias = [True, False]
        for mode, b in itertools.product(modes, bias):
            num = 0
            loc = os.path.dirname(os.path.abspath(__file__))
            cmd = 'DNNL_VERBOSE=1 python -u {}/fpmath_mode.py --mode="{}" --fpmath="FP32" --bias={}'.format(loc, mode, b)
            with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
                for line in p.stdout.readlines():
                    line = str(line, 'utf-8').strip()
                    if "attr-fpmath:bf16" in line:
                        num = num + 1
            assert num == 0, 'The implicit FP32 to BF16 data type conversion failed to disable.'

    @fpmath_mode_env
    def test_env(self):
        os.environ["IPEX_FP32_MATH_MODE"] = "BF32"
        modes = ["jit", "imperative"]
        bias = [True, False]
        for mode, b in itertools.product(modes, bias):
            num1, num2, num3, num4 = 0, 0, 0, 0
            loc = os.path.dirname(os.path.abspath(__file__))
            cmd = 'DNNL_VERBOSE=1 python -u {}/fpmath_mode.py --mode="{}" --env --bias={}'.format(loc, mode, b)
            with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
                for line in p.stdout.readlines():
                    line = str(line, 'utf-8').strip()
                    if "attr-fpmath:bf16" in line and "convolution" in line:
                        num1 = num1 + 1
                    if "attr-fpmath:bf16" in line and "inner_product" in line:
                        num2 = num2 + 1
                    if "attr-fpmath:bf16" in line and "matmul" in line:
                        num3 = num3 + 1
                    if "attr-fpmath:bf16" in line and "rnn" in line:
                        num4 = num4 + 1
            assert num1 > 0 and num2 > 0 and num3 > 0 and num4 > 0, 'The implicit FP32 to BF16 data type conversion failed to enable.'

if __name__ == '__main__':
    test = unittest.main()
