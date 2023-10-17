import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestTorchMethod(TestCase):
    def test_p2p_copy(self):
        device_count = torch.xpu.device_count()
        print('device_count:', device_count)
        shape = 1000
        data = torch.rand(shape)
        temp = data.clone()
        for i in range(device_count):
            a = torch.zeros(shape, device="xpu:{}".format(i))
            a[:] = temp
            self.assertEqual(data, a.cpu())
            temp = a
