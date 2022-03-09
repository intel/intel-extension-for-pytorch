import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_names(self, dtype=torch.float):
        imgs_dpcpp = torch.rand(2, 3, 5, 7, device="xpu")
        self.assertEqual(False, imgs_dpcpp.has_names())
        self.assertEqual((None, None, None, None), imgs_dpcpp.names)

        imgs_dpcpp = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W')).to(dpcpp_device)
        self.assertEqual(True, imgs_dpcpp.has_names())

        print("imgs_dpcpp names = ", imgs_dpcpp.names)
        self.assertEqual(('N', 'C', 'H', 'W'), imgs_dpcpp.names)

        renamed_imgs_dpcpp = imgs_dpcpp.rename(N='batch', C='channels')
        print("imgs_dpcpp names = ", renamed_imgs_dpcpp.names)
        self.assertEqual(renamed_imgs_dpcpp.names, ('batch', 'channels', 'H', 'W'))

        renamed_imgs_dpcpp = imgs_dpcpp.rename(None)
        print("imgs_dpcpp names = ", renamed_imgs_dpcpp.names)
        self.assertEqual(renamed_imgs_dpcpp.names, (None, None, None, None))

        renamed_imgs_dpcpp = imgs_dpcpp.rename('batch', 'channel', 'height', 'width')
        print("imgs_dpcpp names = ", renamed_imgs_dpcpp.names)
        self.assertEqual(renamed_imgs_dpcpp.names, ('batch', 'channel', 'height', 'width'))
