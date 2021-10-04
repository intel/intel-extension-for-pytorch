import math
import random
import unittest
from functools import reduce
import warnings

import torch
import torch.nn as nn
from torch.fx import GraphModule
import copy

import intel_extension_for_pytorch as ipex

from common_utils import TestCase
from torch.testing import FileCheck

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)

class DropoutTester(TestCase):
    def test_remove_dropout_jit(self):
        model = Net().eval()
        x = torch.randn(2, 3)
        with torch.no_grad():
            trace_model = torch.jit.trace(model, x).eval()
            frozen_mod = torch.jit.freeze(trace_model)
            y = trace_model(x)
            self.assertEqual(x, y)
            FileCheck().check_not("aten::dropout").run(trace_model.graph)

    def test_remove_dropout_fx(self):
        model = Net().eval()
        optimized_model = ipex.optimize(model)
        self.assertTrue(isinstance(optimized_model, GraphModule))
        all_formatted = "\n".join([n.format_node() for n in optimized_model.graph.nodes])
        FileCheck().check_not("dropout").run(all_formatted)
        # disable remove_dropout
        optimized_model = ipex.optimize(model, remove_dropout=False)
        self.assertTrue(isinstance(optimized_model, GraphModule))
        all_formatted = "\n".join([n.format_node() for n in optimized_model.graph.nodes])
        FileCheck().check("dropout").run(all_formatted)


if __name__ == '__main__':
    test = unittest.main()
