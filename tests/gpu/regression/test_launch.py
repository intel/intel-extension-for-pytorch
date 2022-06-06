import time
from torch.testing._internal.common_utils import TestCase


class TestUXMethod(TestCase):
    def test_launch(self):
        start = time.time()
        import intel_extension_for_pytorch
        end = time.time()
        print(f"[ INFO ] Intel Extension for Pytorch Launch Time: {end-start}s")
        self.assertLess((end - start), 10, "launch time great than 10s!")
