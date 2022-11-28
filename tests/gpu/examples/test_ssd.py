import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestTorchMethod(TestCase):
    def test_locations_to_boxes(self, dtype=torch.float):
        """
        Small ops fusion for location box conversion in SSD-MobileNetv1.
        """
        locations = torch.randn([512, 3000, 4]).to("xpu")
        priors = torch.randn([512, 3000, 4]).to("xpu")
        center_variance = 0.1
        size_variance = 0.2

        out = torch.xpu.locations_to_boxes(locations, priors, center_variance, size_variance)

        locations = torch.cat([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], dim=locations.dim() - 1)
        ref = torch.cat([
            locations[..., :2] - locations[..., 2:] / 2,
            locations[..., :2] + locations[..., 2:] / 2
        ], locations.dim() - 1)
        self.assertEqual(out, ref)
