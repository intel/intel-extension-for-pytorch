import torch
import ipex
from torch.testing._internal.common_utils import TestCase

import pytest
import os
import tempfile


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        y = self.conv(x)
        return y

def run_model(level):
    fname = tempfile.mkdtemp() + "_case_verbose"
    stdout = os.dup(1)
    verbose = os.open(fname, os.O_WRONLY | os.O_CREAT)
    os.dup2(verbose, 1)

    m = Module()
    d = torch.rand(1, 1, 112, 112)
    m = m.to('xpu')
    d = d.to('xpu')
    with torch.xpu.onednn_verbose(level):
        m(d)

    os.dup2(stdout, 1)
    os.close(verbose)
    f = open(fname)
    num = 0
    for line in f.readlines():
        if line.strip().startswith("dnnl_verbose"):
            num = num + 1
    f.close()
    os.remove(fname)
    return num > 0

class TestVerbose(TestCase):
    def test_verbose_on(self):
        verb_on = run_model(2)
        assert verb_on, 'oneDNN verbose messages not found.'

    def test_verbose_off(self):
        verb_on = run_model(0)
        assert not verb_on, 'unexpected oneDNN verbose messages found.'
