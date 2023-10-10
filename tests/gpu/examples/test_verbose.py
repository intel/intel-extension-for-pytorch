from torch.testing._internal.common_utils import TestCase

import os
import tempfile


class TestVerbose(TestCase):
    def test_verbose_on(self):
        fname = tempfile.mkdtemp() + "_case_verbose"
        os.environ['ONEDNN_VERBOSE'] = '2'
        cmd = f"""python -c "import torch;import intel_extension_for_pytorch;\
        conv = torch.nn.Conv2d(1, 10, 5, 1).to('xpu');d = torch.rand(1, 1, 112, 112).to('xpu');conv(d)" > {fname}"""
        os.system(cmd)
        del os.environ['ONEDNN_VERBOSE']
        f = open(fname)
        num = 0
        for line in f.readlines():
            if line.strip().startswith("onednn_verbose"):
                num = num + 1
        f.close()
        os.remove(fname)
        verb_on = num > 0
        assert verb_on, "oneDNN verbose messages not found."

    def test_verbose_off(self):
        fname = tempfile.mkdtemp() + "_case_verbose"
        os.environ['ONEDNN_VERBOSE'] = '0'
        cmd = f"""python -c "import torch;import intel_extension_for_pytorch;\
        conv = torch.nn.Conv2d(1, 10, 5, 1).to('xpu');d = torch.rand(1, 1, 112, 112).to('xpu');conv(d)" > {fname}"""
        os.system(cmd)
        del os.environ['ONEDNN_VERBOSE']
        f = open(fname)
        num = 0
        for line in f.readlines():
            if line.strip().startswith("onednn_verbose"):
                num = num + 1
        f.close()
        os.remove(fname)
        verb_on = num > 0
        assert not verb_on, "unexpected oneDNN verbose messages found."
