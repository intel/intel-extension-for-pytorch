import unittest
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

from common_utils import TestCase

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        return self.conv_transpose(x)

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.linear(x)

class Tester(TestCase):
    def _test_conv_check(self, model, input, e_message):
        try:
            m = model.eval()
            m = ipex.optimize(m)
            with torch.no_grad():
                m(input)
                raise RuntimeError("the CHECK doesn't trigger error as expected") 
        except RuntimeError as e:
            self.assertTrue(e_message in str(e))

    def _test_linear_check(self, model, input, e_message):
        try:
            m = model.eval()
            m = ipex.optimize(m, auto_kernel_selection=True)
            with torch.no_grad():
                m(input)
                raise RuntimeError("the CHECK doesn't trigger error as expected") 
        except RuntimeError as e:
            self.assertTrue(e_message in str(e))


    def test_conv_negative_padding(self):
        self._test_conv_check(Conv(16, 33, 3, padding=-1), torch.randn(20, 16, 50, 100), "negative padding is not supported")

    def test_conv_nonpositive_stride(self):
        self._test_conv_check(Conv(16, 33, 3, stride=0), torch.randn(20, 16, 50, 100), "non-positive stride is not supported")

    def test_conv_nonpositive_dilation(self):
        self._test_conv_check(Conv(16, 33, 3, dilation=0), torch.randn(20, 16, 50, 100), "non-positive dilation is not supported")

    def test_conv_input_dims(self):
        self._test_conv_check(Conv(16, 33, 3), torch.randn(20, 16, 50), "Expected 4-dimensional input for 4-dimensional weight [33, 16, 3, 3], but got 3-dimensional input of size [20, 16, 50] instead")

    def test_conv_input_shape(self):
        self._test_conv_check(Conv(16, 33, 3), torch.randn(20, 30, 50, 100), "Given groups=1, weight of size [33, 16, 3, 3], expected input[20, 30, 50, 100] to have 16 channels, but got 30 channels instead")

    def test_conv_kernel_size(self):
        self._test_conv_check(Conv(16, 33, 60), torch.randn(20, 16, 50, 100), "Calculated padded input size per channel: (50 x 100). Kernel size: (60 x 60). Kernel size can't be greater than actual input size")

    def test_conv_transpose_negative_padding(self):
        self._test_conv_check(ConvTranspose(16, 33, 3, padding=-1), torch.randn(20, 16, 50, 100), "negative padding is not supported")

    def test_conv_transpose_nonpositive_stride(self):
        self._test_conv_check(ConvTranspose(16, 33, 3, stride=0), torch.randn(20, 16, 50, 100), "non-positive stride is not supported")

    def test_conv_transpose_nonpositive_dilation(self):
        self._test_conv_check(ConvTranspose(16, 33, 3, dilation=0), torch.randn(20, 16, 50, 100), "non-positive dilation is not supported")

    def test_conv_transpose_input_dims(self):
        self._test_conv_check(ConvTranspose(16, 33, 3), torch.randn(20, 16, 50), "Expected 4-dimensional input for 4-dimensional weight [16, 33, 3, 3], but got 3-dimensional input of size [20, 16, 50] instead")

    def test_conv_transpose_input_shape(self):
        self._test_conv_check(ConvTranspose(16, 33, 3), torch.randn(20, 30, 50, 100), "Given transposed=True, weight of size [16, 33, 3, 3], expected input[20, 30, 50, 100] to have 16 channels, but got 30 channels instead")

    def test_linear(self):
        self._test_linear_check(Linear(16, 33), torch.randn(3), "Check the shapes of mat1 and mat2, they cannot be multiplied!")

    def test_linear_bf16(self):
        self._test_linear_check(Linear(16, 33).to(torch.bfloat16), torch.randn(3).bfloat16(), "Check the shapes of mat1 and mat2, they cannot be multiplied!")

if __name__ == '__main__':
    test = unittest.main()