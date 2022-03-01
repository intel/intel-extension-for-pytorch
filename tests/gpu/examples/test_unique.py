import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

N = 1024
M = 5

class TestTorchMethod(TestCase):

    def test_single_unique(self):
        a_cpu = torch.randint(0, 100, [N], dtype=torch.long)
        a_xpu = a_cpu.xpu()

        output_cpu = torch.unique(a_cpu, sorted=True)
        output_xpu = torch.unique(a_xpu, sorted=True)
        self.assertEqual(output_cpu, output_xpu)

        output_cpu = torch.unique(a_cpu, sorted=False)
        output_xpu = torch.unique(a_xpu, sorted=False)

        self.assertTrue(len(output_cpu) == len(output_xpu))
        list_output_cpu = list(output_cpu.numpy())
        list_output_xpu = list(output_xpu.cpu().numpy())

        for item in list_output_cpu:
            self.assertTrue(item in list_output_xpu)
        for item in list_output_xpu:
            self.assertTrue(item in list_output_cpu)


    def test_inverse_unique(self):
        a_cpu = torch.randint(0, 100, [N], dtype=torch.long)
        a_xpu = a_cpu.xpu()

        output_cpu, inverse_cpu = torch.unique(a_cpu, sorted=True, return_inverse=True)
        output_xpu, inverse_xpu = torch.unique(a_xpu, sorted=True, return_inverse=True)
        self.assertEqual(output_cpu, output_xpu)
        self.assertEqual(inverse_cpu, inverse_xpu)


    def test_inverse_counts_unique(self):
        a_cpu = torch.randint(0, 100, [N], dtype=torch.long)
        a_xpu = a_cpu.xpu()

        output_cpu, inverse_cpu, counts_cpu = torch.unique(a_cpu, sorted=True, return_inverse=True, return_counts=True)
        output_xpu, inverse_xpu, counts_xpu = torch.unique(a_xpu, sorted=True, return_inverse=True, return_counts=True)
        self.assertEqual(output_cpu, output_xpu)
        self.assertEqual(inverse_cpu, inverse_xpu)
        self.assertEqual(counts_cpu, counts_xpu)


    def test_dim0_unique(self):
        a_cpu = torch.randint(0, 5, [N, M], dtype=torch.long)
        a_xpu = a_cpu.xpu()

        output_cpu, inverse_cpu, counts_cpu = torch.unique(a_cpu, sorted=True, return_inverse=True, return_counts=True, dim=0)
        output_xpu, inverse_xpu, counts_xpu = torch.unique(a_xpu, sorted=True, return_inverse=True, return_counts=True, dim=0)
        self.assertEqual(output_cpu, output_xpu)
        self.assertEqual(inverse_cpu, inverse_xpu)
        self.assertEqual(counts_cpu, counts_xpu)


    def test_dim1_unique(self):
        a_cpu = torch.randint(0, 5, [N, M], dtype=torch.long)
        a_xpu = a_cpu.xpu()

        output_cpu, inverse_cpu, counts_cpu = torch.unique(a_cpu, sorted=True, return_inverse=True, return_counts=True, dim=1)
        output_xpu, inverse_xpu, counts_xpu = torch.unique(a_xpu, sorted=True, return_inverse=True, return_counts=True, dim=1)
        self.assertEqual(output_cpu, output_xpu)
        self.assertEqual(inverse_cpu, inverse_xpu)
        self.assertEqual(counts_cpu, counts_xpu)

    def test_corner_unique(self):
        # this case is aimed to check https://jira.devtools.intel.com/browse/PYTORCHDGQ-1225
        a_cpu = torch.tensor([i for i in range(2048)])
        a_cpu[1023] = 1022
        a_cpu[2047] = 2046
        a_xpu = a_cpu.xpu()

        output_cpu, inverse_cpu, counts_cpu = torch.unique(a_cpu, sorted=True, return_inverse=True, return_counts=True)
        output_xpu, inverse_xpu, counts_xpu = torch.unique(a_xpu, sorted=True, return_inverse=True, return_counts=True)
        self.assertEqual(output_cpu, output_xpu)
        self.assertEqual(inverse_cpu, inverse_xpu)
        self.assertEqual(counts_cpu, counts_xpu)
