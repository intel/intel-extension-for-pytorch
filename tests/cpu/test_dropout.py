import unittest

import torch
import torch.nn as nn

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

    def test_replace_dropout_with_identity(self):
        model = Net().eval()
        optimized_model = ipex.optimize(model)
        x = torch.randn(2, 3)
        named_children = dict(optimized_model.named_children())
        self.assertTrue(isinstance(named_children["dropout"], torch.nn.Identity))

        optimized_model = ipex.optimize(model, replace_dropout_with_identity=False)
        named_children = dict(optimized_model.named_children())
        self.assertTrue(isinstance(named_children["dropout"], torch.nn.Dropout))


if __name__ == "__main__":
    test = unittest.main()
