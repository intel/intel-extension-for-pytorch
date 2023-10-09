import torch
from torch.testing._internal.common_utils import TestCase, freeze_rng_state

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):

    def test_generator_xpu(self):
        # tests Generator API
        # manual_seed, seed, initial_seed, get_state, set_state
        g1 = torch.Generator(device='xpu')
        g2 = torch.Generator(device='xpu')
        g3 = torch.Generator(device='xpu')
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        g3.manual_seed(12345)

        self.assertEqual(g1.initial_seed(), g2.initial_seed())
        self.assertEqual(g1.initial_seed(), g3.initial_seed())
        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        # cannot set cpu_generator_state to xpu_generator,  len(cpu_generator_state) is 5048, for xpu is 16.
        g2_state = g2.get_state()
        # must add param device, or it will hang
        g2_randn = torch.randn(1, generator=g2, device='xpu')
        g1.set_state(g2_state)
        g1_randn = torch.randn(1, generator=g1, device='xpu')
        self.assertEqual(g1_randn, g2_randn)


    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().xpu()
            torch.xpu.manual_seed(2)
            self.assertEqual(torch.xpu.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.xpu.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.xpu.initial_seed(), 2)
