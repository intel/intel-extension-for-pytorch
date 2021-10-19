import unittest, copy
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import time, sys
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from test_jit_llga_quantization_fuser import llga_test_env
import torchvision.models as models
import torch.fx.experimental.optimization as optimization

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

class SimpleNet_v2(torch.nn.Module):
    def __init__(self):
        super(SimpleNet_v2, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.conv2(x1)
        y = torch.flatten(x1, start_dim=1)
        return y

class TestCoreBinding(TestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_decorator_function_result(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        cpu_pool = ipex.cpu.runtime.CPUPool([1, 2, 3, 4])
        @ipex.cpu.runtime.pin(cpu_pool)
        def test(model, x):
            return model(x)

        y_runtime = test(model, x)

        y = model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_with_function_result(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        cpu_pool = ipex.cpu.runtime.CPUPool([1, 2, 3, 4])
        with ipex.cpu.runtime.pin(cpu_pool):
            y_runtime = model(x)

        y = model(x)
        self.assertEqual(y, y_runtime)

class TestRuntimeAPI(TestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_module_result(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(model, cpu_pool)

        # Task submit and wait
        y_runtime_future = task(x)
        y_runtime = y_runtime_future.get()
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_sync_task_result(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(model, cpu_pool)

        # Task sync submit
        y_runtime = task.run_sync(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_function_result(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        def test(model, x):
            return model(x)
        # Calculate the reference result
        y = test(model, x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(test, cpu_pool)

        # Task submit and wait
        y_runtime_future = task(model, x)
        y_runtime = y_runtime_future.get()
        self.assertEqual(y, y_runtime)

if __name__ == '__main__':
    test = unittest.main()
