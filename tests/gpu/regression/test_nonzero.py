import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

class TestTorchMethod(TestCase):
    @pytest.mark.skip(reason="Unstable outputs on ATS-P")
    def test_nonzero_memory_leak(self):
        '''
        Regression desc:
            nonzero may cause memory leak when the number of nonzeros
            is not a multiple of work group size
        '''
        B = 4
        N = 15000
        C = 3
        npoint = 512

        torch.xpu.manual_seed(0)
        xyz = torch.rand(B, N, C, device='xpu')
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.ones(B, N, dtype=xyz.dtype, device=xyz.device) * 1e10
        batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

        for i in range(npoint):
            print("Iter #", i)
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance

            distance_cpu = distance.clone().cpu()

            distance[mask] = dist[mask]
            distance_cpu[mask] = dist.cpu()[mask.cpu()]

            print('CPU distance: ', distance_cpu)
            print('XPU distance: ', distance.cpu())

            self.assertEqual(distance, distance_cpu)

            farthest = torch.max(distance, -1)[1]
