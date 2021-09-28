import unittest, copy
import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

class CPUOPsTester(TestCase):

    def test_channelshuffle(self):
        channel_shuffle = torch.nn.ChannelShuffle(20)
        x = torch.randn(3, 40, 20, 20)
        x1 = x.clone()
        y1 = channel_shuffle(x1)

        # test channels last
        x2 = x.clone().to(memory_format=torch.channels_last)
        y2 = channel_shuffle(x2)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)

    def test_pixelshuffle(self):
        pixel_shuffle = torch.nn.PixelShuffle(30)
        x = torch.randn(3, 900, 40, 40)
        x1 = x.clone()
        y1 = pixel_shuffle(x1)

        # test channels last
        x2 = x.clone().to(memory_format=torch.channels_last)
        y2 = pixel_shuffle(x2)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)

if __name__ == '__main__':
    test = unittest.main()
