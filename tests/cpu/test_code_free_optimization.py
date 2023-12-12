import unittest
from common_utils import TestCase
import os
import subprocess
import itertools
import torch
import logging

logging.getLogger().setLevel(logging.DEBUG)


class TestCodeFreeOptimization(TestCase):
    def test_conv_bn(self):
        loc = os.path.dirname(os.path.abspath(__file__))
        disable_ipex_graph_modes = [False, True]
        dtypes = (
            ["float32", "bfloat16"]
            if torch.ops.mkldnn._is_mkldnn_bf16_supported()
            else ["float32"]
        )
        for disable_ipex_graph_mode, dtype in itertools.product(
            disable_ipex_graph_modes, dtypes
        ):
            _ipex_optimize_hit_count = 0
            _ipex_convolution = False
            _has_batchnorm = False
            cmd = "ipexrun --ninstances 1 "
            cmd += "--auto-ipex "
            cmd += "--dtype {} ".format(dtype)
            cmd += "--auto-ipex-verbose "
            if disable_ipex_graph_mode:
                cmd += "--disable-ipex-graph-mode "
            cmd += "{}/code_free_optimization.py --conv_bn".format(loc)
            with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as p:
                for line in p.stdout.readlines():
                    line = str(line, "utf-8").strip()
                    if line.__contains__("_ipex_optimize_hit_count"):
                        _ipex_optimize_hit_count = _ipex_optimize_hit_count + 1
                    if line.__contains__(
                        "torch_ipex::convolution_forward_impl"
                        if disable_ipex_graph_mode
                        else "ipex_prepack::convolution_run"
                    ) or line.__contains__("fused_to"):
                        _ipex_convolution = True
                    if line.__contains__("batch_norm"):
                        _has_batchnorm = True
            assert (
                _ipex_optimize_hit_count == 1
            ), "Expect hit once of ipex.optimize globally"
            assert _ipex_convolution, "Expect use ipex convolution by ipex.optimize"
            assert _has_batchnorm is False, "should not see bn"

    def test_conv_bn_with_module_created_in_forward(self):
        loc = os.path.dirname(os.path.abspath(__file__))
        disable_ipex_graph_modes = [False, True]
        dtypes = (
            ["float32", "bfloat16"]
            if torch.ops.mkldnn._is_mkldnn_bf16_supported()
            else ["float32"]
        )
        for disable_ipex_graph_mode, dtype in itertools.product(
            disable_ipex_graph_modes, dtypes
        ):
            _ipex_optimize_hit_count = 0
            _ipex_convolution = False
            cmd = "ipexrun --ninstances 1 "
            cmd += "--auto-ipex "
            cmd += "--dtype {} ".format(dtype)
            cmd += "--auto-ipex-verbose "
            if disable_ipex_graph_mode:
                cmd += "--disable-ipex-graph-mode "
            cmd += "{}/code_free_optimization.py --conv_bn_with_module_created_in_forward".format(
                loc
            )
            with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as p:
                for line in p.stdout.readlines():
                    line = str(line, "utf-8").strip()
                    if line.__contains__("_ipex_optimize_hit_count"):
                        _ipex_optimize_hit_count = _ipex_optimize_hit_count + 1
                    if line.__contains__(
                        "torch_ipex::convolution_forward_impl"
                        if disable_ipex_graph_mode
                        else "ipex_prepack::convolution_run"
                    ) or line.__contains__("fused_to"):
                        _ipex_convolution = True
            assert (
                _ipex_optimize_hit_count == 1
            ), "Expect hit once of ipex.optimize globally"
            assert _ipex_convolution, "Expect use ipex convolution by ipex.optimize"
            # Not check BN, because FX limitation, ipex.optimize failed to do fusion

    def test_auto_ipex_module(self):
        loc = os.path.dirname(os.path.abspath(__file__))
        disable_ipex_graph_modes = [False, True]
        dtypes = (
            ["float32", "bfloat16"]
            if torch.ops.mkldnn._is_mkldnn_bf16_supported()
            else ["float32"]
        )
        for disable_ipex_graph_mode, dtype in itertools.product(
            disable_ipex_graph_modes, dtypes
        ):
            _ipex_optimize_hit_count = 0
            _ipex_convolution = False
            _has_batchnorm = False
            cmd = "python -m intel_extension_for_pytorch.cpu.auto_ipex "
            cmd += "--dtype {} ".format(dtype)
            cmd += "--auto-ipex-verbose "
            if disable_ipex_graph_mode:
                cmd += "--disable-ipex-graph-mode "
            cmd += "{}/code_free_optimization.py --conv_bn".format(loc)
            with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as p:
                for line in p.stdout.readlines():
                    line = str(line, "utf-8").strip()
                    if line.__contains__("_ipex_optimize_hit_count"):
                        _ipex_optimize_hit_count = _ipex_optimize_hit_count + 1
                    if line.__contains__(
                        "torch_ipex::convolution_forward_impl"
                        if disable_ipex_graph_mode
                        else "ipex_prepack::convolution_run"
                    ) or line.__contains__("fused_to"):
                        _ipex_convolution = True
                    if line.__contains__("batch_norm"):
                        _has_batchnorm = True
            assert (
                _ipex_optimize_hit_count == 1
            ), "Expect hit once of ipex.optimize globally"
            assert _ipex_convolution, "Expect use ipex convolution by ipex.optimize"
            assert _has_batchnorm is False, "should not see bn"


if __name__ == "__main__":
    test = unittest.main()
