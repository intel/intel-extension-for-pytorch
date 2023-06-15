import unittest
from common_utils import TestCase
import os
import subprocess

class TestConvReorder(TestCase):
    def test_conv_with_itensor_size1(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen('DNNL_VERBOSE=1 python  -u {}/itensor_size1_test.py'.format(loc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.__contains__("reorder"):
                    num = num + 1
        assert num == 3 , 'conv channelslast has unexpected reorder.'


if __name__ == '__main__':
    test = unittest.main()




