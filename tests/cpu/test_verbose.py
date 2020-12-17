import unittest
from common_utils import VerboseTestCase
import subprocess

class TestLinearWeightPack(VerboseTestCase):
    def test_linear_weight_pack(self):
        with subprocess.Popen('DNNL_VERBOSE=1 python -u linear_prepack.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            retval = p.wait()
            segmentation = {
                'fp32': {'reorder_for_pack': 1},
                'auto-mix for bf16': {'reorder_for_pack': 1, 'reorder_for_dtype':5},
                'back to fp32':  {'reorder_for_pack': 2, 'reorder_for_dtype':4},
                'auto-mix for int8': {'reorder_for_pack': 1, 'reorder_for_dtype':5},
            }
            seg = None
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.endswith('***************'):
                    seg = line.strip().split(',')[0]
                    continue
                if self.is_dnnl_verbose(line) and self.ReorderForPack(line):
                    segmentation[seg]['reorder_for_pack'] -= 1
                    self.assertTrue(segmentation[seg]['reorder_for_pack'] >=0, "show unexpected reorder for pack")
                if self.is_dnnl_verbose(line) and self.OnlyReorderDtype(line):
                    self.assertTrue(segmentation[seg]['reorder_for_dtype'] >=0, "show unexpected reorder for dtype")
                    segmentation[seg]['reorder_for_dtype'] -= 1

if __name__ == '__main__':
    test = unittest.main()
