import unittest, copy
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import time, sys
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
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

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_nested_with_function_result(self):
        model = torch.nn.Softmax(dim=-1)
        model.eval()
        x = torch.rand(100, 8276)
        cpu_pool = ipex.cpu.runtime.CPUPool([1, 2])
        cpu_pool2 = ipex.cpu.runtime.CPUPool([3, 4])
        with ipex.cpu.runtime.pin(cpu_pool):
            y_runtime = model(x)
            with ipex.cpu.runtime.pin(cpu_pool2):
                y_runtime = model(x)
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

class TestMultiStreamModule(TestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_module(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=2, cpu_pool=cpu_pool)

        y_runtime = multi_stream_model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_single_stream(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=1, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(model, num_streams=1, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, y_runtime2[0])

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_core_number_not_divisible_stream_number(self):
        model = SimpleNet()
        model.eval()
        num_streams = 2
        batch_size = num_streams
        x = torch.rand(batch_size, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        # Core Number is 3, stream Number is 2
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_batchsize_less_than_stream_number(self):
        model = SimpleNet()
        model.eval()
        num_streams = 3
        batch_size = 2
        x = torch.rand(batch_size, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        # Batchsize 2, Core Number is 3, stream Number is 3
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_batchsize_not_divisible_stream_number(self):
        model = SimpleNet()
        model.eval()
        num_streams = 3
        batch_size = 4
        x = torch.rand(batch_size, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        # Batchsize 4, Core Number is 3, stream Number is 3
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))
        self.assertEqual(y_runtime2[0].size(0), 2)
        self.assertEqual(y_runtime2[1].size(0), 1)
        self.assertEqual(y_runtime2[2].size(0), 1)

if __name__ == '__main__':
    test = unittest.main()
