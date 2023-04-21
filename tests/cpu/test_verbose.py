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
                # TODO: with oneDNN commit bb4ed4b, ONEDNN_VERBOSE=0 still gives several lines of logs on
                # ONEDNN_VERBOSE,info... Workaround by checking the exec and create logs to
                # unblock the oneDNN upgrade in IPEX. Will revert this change back once fixed
                # by oneDNN.
                if line.startswith(("onednn_verbose,exec", "onednn_verbose,create")):
                    num = num + 1
        assert num == 0, 'unexpected oneDNN verbose messages found.'

if __name__ == '__main__':
    test = unittest.main()
