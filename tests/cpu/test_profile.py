import unittest
from common_utils import TestCase
import os
import subprocess

class TestProfiler(TestCase):
    #currently only check ipex softmax as an example
    def test_profile_on(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen('python  -u {}/profile_ipex_op.py'.format(loc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.__contains__("dil_softmax_"):
                    num = num + 1
        assert num == 2 , 'IPEX op profiling info not found.'


if __name__ == '__main__':
    test = unittest.main()



