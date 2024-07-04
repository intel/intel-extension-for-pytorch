import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa
ipex.compatible_mode()

cuda_device = torch.device("cuda")


class TestTorchMethod(TestCase):
    def test_data_loader(self):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                "Dataset Initialization"
                pass

            def __len__(self):
                return 16

            def __getitem__(self, index):
                x = torch.ones((5, 6))
                return x

        dataset = Dataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=256, pin_memory=True, pin_memory_device="cuda"
        )

        for _, sample in enumerate(loader):
            sample = sample.cuda()
            self.assertEqual(sample.device.type, "xpu")
