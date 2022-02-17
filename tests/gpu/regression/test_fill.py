import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex
import pytest
import numpy as np

class TestFill(TestCase):
    def test_fill(self):
        '''
        Regression desc:
          fill_ may set values to part of large-size tensor.
        '''
        output = torch.randn((2, 3136, 218089)).xpu()
        output_cpu = output.to("cpu")
        output.fill_(2.22)
        output_cpu.fill_(2.22)
        self.assertEqual(output.to("cpu"), output_cpu)
