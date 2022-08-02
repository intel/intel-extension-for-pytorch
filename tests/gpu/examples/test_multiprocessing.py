import os

import torch
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import (TestCase, IS_WINDOWS, TEST_WITH_TSAN)
import intel_extension_for_pytorch

import pytest


def queue_get_exception(inqueue, outqueue):
    os.close(2)  # hide expected error message
    try:
        torch.zeros(5, 5).to('xpu')
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put('no exception')


@pytest.mark.skipif(TEST_WITH_TSAN, reason="TSAN is not fork-safe since we're forking in a multi-threaded environment")
class TestTorchMethod(TestCase):

    def tearDown(self):
        # Temporarily missing a function that keep tests isolated from each-other. 
        pass

    @pytest.mark.skipif(IS_WINDOWS, reason='not applicable to Windows (only fails with fork)')
    def test_xpu_bad_call(self):
        # Initialize XPU
        t = torch.zeros(5, 5).to('xpu')
        inq = mp.Queue()
        outq = mp.Queue()
        p = mp.Process(target=queue_get_exception, args=(inq, outq))
        p.start()
        inq.put(t)
        p.join()
        self.assertIsInstance(outq.get(), RuntimeError)

    @pytest.mark.skip(reason='tearDown is not available, so only test bad fork case.')
    @pytest.mark.skipif(IS_WINDOWS, reason='not applicable to Windows (only fails with fork)')
    def test_good_xpu_fork(self):
        size = 2
        processes = []
        for _ in range(size):
            q = mp.Queue()
            p = mp.Process(target=queue_get_exception, args=(None, q))
            processes.append((q, p))
            p.start()
        for q, p in processes:
            p.join()
            self.assertEqual(q.get(), 'no exception')

    @pytest.mark.skipif(IS_WINDOWS, reason='not applicable to Windows (only fails with fork)')
    def test_wrong_xpu_fork(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
import torch
import intel_extension_for_pytorch
from torch.multiprocessing import Process
def run():
    a = torch.rand(10, 1).to('xpu')
if __name__ == "__main__":
    size = 2
    processes = []
    # it would work fine without the line below
    x = torch.rand(20, 2).to('xpu')
    for _ in range(size):
        p = Process(target=run)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
""")
        self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")
