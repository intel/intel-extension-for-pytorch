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

class TestJitRuntimeAPI(JitTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_trace_module(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        # Calculate the reference result
        trace_model = torch.jit.trace(model, x)
        y = trace_model(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(trace_model, cpu_pool)

        # Task submit and get
        y_runtime_future = task(x)
        y_runtime = y_runtime_future.get()
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_sync_trace_module(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        # Calculate the reference result
        trace_mode = torch.jit.trace(model, x)
        y = trace_mode(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(trace_mode, cpu_pool)

        # Task sync run
        y_runtime = task.run_sync(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_bf16_task_core_bind(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_mode = torch.jit.trace(model, x)
        y = trace_mode(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(trace_mode, cpu_pool)

        # Task submit and wait
        y_runtime_future = task(x)
        y_runtime = y_runtime_future.get()
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_task_multi_submission(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_mode = torch.jit.trace(model, x)
        y = trace_mode(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(trace_mode, cpu_pool)

        # Submit task 3 times, then wait for result
        y_runtime = []
        y_runtime_future = []
        for i in range(3):
            y_runtime_future.append(task(x))
        for item in y_runtime_future:
            y_runtime.append(item.get())

        self.assertEqual(y, y_runtime[0])
        self.assertEqual(y, y_runtime[1])
        self.assertEqual(y, y_runtime[2])

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_bf16_module(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_model = torch.jit.trace(model, x)
        y = trace_model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=2, cpu_pool=cpu_pool)

        y_runtime = multi_stream_model(x)
        self.assertEqual(y, y_runtime)

class TestLLGARuntimeAPI(JitLlgaTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    @llga_test_env
    def test_int8_simpleNet_task_core_bind(self):
        with torch.no_grad():
            model = SimpleNet_v2()
            model.eval()
            x = torch.rand(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)

            # Calculate the reference result
            graph, m_llga, m_cpu = self.prepareModel(model, [x], folding=True, qscheme=torch.per_tensor_symmetric)
            y = m_llga(x)

            # Create task
            cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            task = ipex.cpu.runtime.Task(m_llga, cpu_pool)

            # Task submit and wait
            y_runtime_future = task(x)
            y_runtime = y_runtime_future.get()
            self.assertEqual(y, y_runtime)

if __name__ == '__main__':
    test = unittest.main()
