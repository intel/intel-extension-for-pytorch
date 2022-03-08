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

class TestJitRuntimeAPI(JitTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_task_async_api_fp32_jit_model(self):
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
    def test_task_sync_api_fp32_jit_model(self):
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
    def test_task_async_api_bf16_jit_model(self):
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
    def test_task_async_api_bf16_jit_model_multi_submission(self):
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
    def test_task_copy_bf16_jit_mode(self):
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

        # Copy Task
        task2 = task

        # Task submit and wait
        y_runtime_future = task(x)
        y_runtime = y_runtime_future.get()
        y_runtime_future2 = task2(x)
        y_runtime2 = y_runtime_future2.get()
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, y_runtime2)

class TestJITMultiStreamModule(JitTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_module_bf16_jit_model(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)
        num_streams = batch_size

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_model = torch.jit.trace(model, x)
        y = trace_model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=num_streams, cpu_pool=cpu_pool)

        y_runtime = multi_stream_model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_module_bf16_jit_model_concat_output(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)
        num_streams = batch_size

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_model = torch.jit.trace(model, x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=num_streams, cpu_pool=cpu_pool)
        y_runtime = multi_stream_model(x)

        # Create MultiStreamModule with concat_output=False
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y_runtime2.__len__(), num_streams)
        self.assertEqual(y_runtime, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_single_stream_module_bf16_jit_model(self):
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
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=1, cpu_pool=cpu_pool)
        y_runtime = multi_stream_model(x)

        # Create MultiStreamModule with concat_output=False
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(trace_model, num_streams=1, cpu_pool=cpu_pool, concat_output=False)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, y_runtime2[0])

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_core_number_not_divisible_stream_number_bf16_jit_model(self):
        model = SimpleNet()
        model.eval()
        num_streams = 2
        batch_size = num_streams
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)
        
        # Calculate the reference result
        y = traced_model(x)

        # Create MultiStreamModule
        # Core Number is 3, stream Number is 2
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_batchsize_less_than_stream_number_bf16_jit_model(self):
        model = SimpleNet()
        model.eval()
        num_streams = 3
        batch_size = 2
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)

        # Calculate the reference result
        y = traced_model(x)

        # Create MultiStreamModule
        # Batchsize 2, Core Number is 3, stream Number is 3
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_batchsize_not_divisible_stream_number_bf16_jit_model(self):
        model = SimpleNet()
        model.eval()
        num_streams = 3
        batch_size = 4
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)

        # Calculate the reference result
        y = traced_model(x)

        # Create MultiStreamModule
        # Batchsize 4, Core Number is 3, stream Number is 3
        cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool)
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(traced_model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)

        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))
        self.assertEqual(y_runtime2[0].size(0), 2)
        self.assertEqual(y_runtime2[1].size(0), 1)
        self.assertEqual(y_runtime2[2].size(0), 1)

class TestLLGARuntimeAPI(JitLlgaTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_task_async_api_int8_jit_model(self):
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

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_module_int8_jit_model(self):
        with torch.no_grad():
            model = SimpleNet_v2()
            model.eval()
            x = torch.rand(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)

            # Calculate the reference result
            graph, m_llga, m_cpu = self.prepareModel(model, [x], folding=True, qscheme=torch.per_tensor_symmetric)
            y = m_llga(x)

            # Create task
            cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
            multi_stream_model = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=1, cpu_pool=cpu_pool)
            multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=1, cpu_pool=cpu_pool, concat_output=False)

            # Task submit and wait
            y_runtime = multi_stream_model(x)
            y_runtime2 = multi_stream_model2(x)
            self.assertEqual(y, y_runtime)
            self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_core_number_not_divisible_stream_number_int8_jit_model(self):
        with torch.no_grad():
            model = SimpleNet_v2()
            model.eval()
            num_streams = 2
            batch_size = num_streams
            x = torch.rand(batch_size, 3, 16, 16).contiguous(memory_format=torch.channels_last)

            # Calculate the reference result
            graph, m_llga, m_cpu = self.prepareModel(model, [x], folding=True, qscheme=torch.per_tensor_symmetric)
            y = m_llga(x)

            # Create MultiStreamModule
            # Core Number is 3, stream Number is 2
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
            multi_stream_model = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=num_streams, cpu_pool=cpu_pool)
            multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

            # Task submit and wait
            y_runtime = multi_stream_model(x)
            y_runtime2 = multi_stream_model2(x)
            self.assertEqual(y, y_runtime)
            self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_batchsize_less_than_stream_number_int8_jit_model(self):
        with torch.no_grad():
            model = SimpleNet_v2()
            model.eval()
            num_streams = 3
            batch_size = 2
            x = torch.rand(batch_size, 3, 16, 16).contiguous(memory_format=torch.channels_last)

            # Calculate the reference result
            graph, m_llga, m_cpu = self.prepareModel(model, [x], folding=True, qscheme=torch.per_tensor_symmetric)
            y = m_llga(x)

            # Create MultiStreamModule
            # Batchsize is 2, Core Number is 3, stream Number is 3
            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=[0, 1, 2])
            multi_stream_model = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=num_streams, cpu_pool=cpu_pool)
            multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(m_llga, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False)

            # Task submit and wait
            y_runtime = multi_stream_model(x)
            y_runtime2 = multi_stream_model2(x)
            self.assertEqual(y, y_runtime)
            self.assertEqual(y, torch.cat(y_runtime2))
            self.assertEqual(y_runtime2.__len__(), batch_size)

if __name__ == '__main__':
    test = unittest.main()
