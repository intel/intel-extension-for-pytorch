import unittest
from common_utils import TestCase
import os
import subprocess

class TestVerbose(TestCase):
    def test_verbose_on(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen('python -u {}/verbose.py --verbose-level=1'.format(loc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
        assert num > 0, 'oneDNN verbose messages not found.'

    def test_verbose_off(self):
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen('python -u {}/verbose.py --verbose-level=0'.format(loc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
        assert num == 0, 'unexpected oneDNN verbose messages found.'

if __name__ == '__main__':
    test = unittest.main()
