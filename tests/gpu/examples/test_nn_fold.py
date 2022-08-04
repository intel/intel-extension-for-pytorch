import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_nn_flod(self, dtype=torch.float):
        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        fold_input = torch.randn(1, 3 * 2 * 2, 12)
        output_flod = fold(fold_input)
        # print(output)

        unfold = nn.Unfold(kernel_size=(2, 3))
        unfold_input = torch.randn(2, 5, 3, 4)
        output_unflod = unfold(unfold_input)
        # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
        # 4 blocks (2x3 kernels) in total in the 3x4 input
        # print(output)

        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        inp = torch.randn(1, 3, 10, 12)
        w = torch.randn(2, 3, 4, 5)
        inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        out_unf = inp_unf.transpose(1, 2).matmul(
            w.view(w.size(0), -1).t()).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        # or equivalently (and avoiding a copy),
        # out = out_unf.view(1, 2, 7, 8)
        print((torch.nn.functional.conv2d(inp, w) - out).abs().max())

        fold_dpcpp = fold.to("xpu")
        fold_input_dpcpp = fold_input.to("xpu")
        output_dpcpp_flod = fold_dpcpp(fold_input_dpcpp)
        # print(output_dpcpp.to("cpu"))

        unfold_dpcpp = unfold.to("xpu")
        unfold_input_dpcpp = unfold_input.to("xpu")
        output_dpcpp_unflod = unfold_dpcpp(unfold_input_dpcpp)
        # print(output_dpcpp.to("cpu"))
        self.assertEqual(output_flod, output_dpcpp_flod.cpu())
        self.assertEqual(output_unflod, output_dpcpp_unflod.cpu())
