import unittest, copy
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import time, sys
from test_ao_jit_llga_utils import JitLlgaTestCase
import torch.fx.experimental.optimization as optimization
from test_runtime_api import TestInputOutputModule, TestInputOutputModule2

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

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_stream_number_auto_bf16_jit_model(self):
        model = torch.nn.Softmax(dim=-1)
        model.eval()
        for i in range(ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()):
            batch_size = list(range(i+1)).__len__()
            x = torch.rand(batch_size, 64)

            # Calculate the reference result
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            traced_model = torch.jit.freeze(traced_model)

            # Warm Up
            for i in range(3):
                traced_model(x)

            # Calculate the reference result
            y = traced_model(x)

            cpu_pool = ipex.cpu.runtime.CPUPool(core_ids=list(range(i+1)))

            # The stream number will be determined automatically.
            multi_stream_model = ipex.cpu.runtime.MultiStreamModule(traced_model, cpu_pool=cpu_pool)
            y_runtime = multi_stream_model(x)
            stream_num_ground_truth = ipex.cpu.runtime.get_default_num_streams(cpu_pool)
            self.assertEqual(y, y_runtime)
            self.assertEqual(multi_stream_model.get_stream_number(), stream_num_ground_truth)

class TestLLGARuntimeAPI(JitLlgaTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_task_async_api_int8_jit_model(self):
        with torch.no_grad():
            model = SimpleNet_v2()
            model.eval()
            x = torch.rand(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)

            # Calculate the reference result
            graph, m_llga, m_cpu = self.prepareModel(model, [x])
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
            graph, m_llga, m_cpu = self.prepareModel(model, [x])
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
            graph, m_llga, m_cpu = self.prepareModel(model, [x])
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
            graph, m_llga, m_cpu = self.prepareModel(model, [x])
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

class TestMultiStreamModuleHint(JitTestCase):
    def init_set_up(self):
        # Create Multi Stream Module without concat output
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        num_streams = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        return batch_size, num_streams, cpu_pool

    def create_jit_traced_model(self, model, input):
        traced_model = torch.jit.trace(model, input).eval()
        traced_model = torch.jit.freeze(traced_model)
        return traced_model

    def create_multi_stream_module(self,
                                traced_model,
                                num_streams,
                                cpu_pool,
                                multi_stream_input_hint,
                                multi_stream_output_hint = None,
                                concat_output = True):

        if not concat_output:
            return ipex.cpu.runtime.MultiStreamModule(traced_model,
                                                    num_streams=num_streams,
                                                    cpu_pool=cpu_pool,
                                                    concat_output = False,
                                                    input_split_hint = multi_stream_input_hint)

        else:
            return ipex.cpu.runtime.MultiStreamModule(traced_model,
                                                    num_streams=num_streams,
                                                    cpu_pool=cpu_pool,
                                                    input_split_hint = multi_stream_input_hint,
                                                    output_concat_hint = multi_stream_output_hint)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_input_output_hint(self):
        batch_size, num_streams, cpu_pool = self.init_set_up()

        # This module:
        #   * Accept 3 tensors as input
        #   * Return a tuple of 3 tensors as output
        model = TestInputOutputModule().eval()
        for batch_size in (num_streams-1, num_streams):
            # There is test for when batch_size is less than num_streams
            input_tensor1 = torch.rand(batch_size, 1)
            input_tensor2 = torch.rand(batch_size, 1)
            input_tensor3 = torch.rand(batch_size, 1)

            # Since jit trace only accept single tensor or a tuple of tensors as input
            # https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace
            jit_input = (input_tensor1, input_tensor2, input_tensor3)

            traced_model = self.create_jit_traced_model(model, jit_input)

            # Warm Up in the main thread to finish the jit pass optimizations
            for _ in range(3):
                traced_model(input_tensor1, input_tensor2, input_tensor3)

            # Calculate the reference result
            y_ref = traced_model(input_tensor1, input_tensor2, input_tensor3)

            multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(0, 0, 0)

            multi_stream_model = self.create_multi_stream_module(traced_model,
                                                                num_streams,
                                                                cpu_pool,
                                                                multi_stream_input_hint,
                                                                concat_output=False)
            y_runtime = multi_stream_model(input_tensor1, input_tensor2, input_tensor3)

            # Manually concat the output
            y_runtime_res1 = []
            y_runtime_res2 = []
            y_runtime_res3 = []
            for stream_id in range(num_streams if ((batch_size // num_streams) >= 1) else batch_size):
                y_runtime_res1.append(y_runtime[stream_id][0])
                y_runtime_res2.append(y_runtime[stream_id][1])
                y_runtime_res3.append(y_runtime[stream_id][2])
            y_runtime_res = (torch.cat(y_runtime_res1), torch.cat(y_runtime_res2), torch.cat(y_runtime_res3))
            self.assertEqual(y_ref, y_runtime_res)

            # Create Multi Stream Module with concat output
            multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, 0, 0))

            multi_stream_model2 = self.create_multi_stream_module(traced_model,
                                                                        num_streams,
                                                                        cpu_pool,
                                                                        multi_stream_input_hint,
                                                                        multi_stream_output_hint,
                                                                        concat_output=True)
            y_runtime_res2 = multi_stream_model2(input_tensor1, input_tensor2, input_tensor3)
            self.assertEqual(y_ref, y_runtime_res2)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_simulate_bert_large_input_output(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, key1, key2, key3):
                return key1*2, key2*2
        # This module simulates the behaviour of Bert Large LZ models:
        #   * Accept 3 tensors (with key word) as input
        #   * Return a tuple of 2 tensors as output
        model = TestModule().eval()

        batch_size, num_streams, cpu_pool = self.init_set_up()
        jit_input = (torch.rand(batch_size, 1), torch.rand(batch_size, 2), torch.rand(batch_size, 3))
        traced_model = self.create_jit_traced_model(model, jit_input)

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 1)
        input_tensor3 = torch.rand(batch_size, 1)

        # Warm Up
        for _ in range(3):
            traced_model(key1=input_tensor1, key2=input_tensor2, key3=input_tensor3)

        # Calculate the reference result
        y_ref = traced_model(key1=input_tensor1, key2=input_tensor2, key3=input_tensor3)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(key1=0, key2=0, key3=0)

        multi_stream_model = self.create_multi_stream_module(traced_model,
                                                            num_streams,
                                                            cpu_pool,
                                                            multi_stream_input_hint,
                                                            concat_output=False)
        y_runtime = multi_stream_model(key1=input_tensor1, key2=input_tensor2, key3=input_tensor3)

        # Manually Concat the output
        y_runtime_res1 = []
        y_runtime_res2 = []
        for i in range(num_streams):
            y_runtime_res1.append(y_runtime[i][0])
            y_runtime_res2.append(y_runtime[i][1])
        y_runtime_res = (torch.cat(y_runtime_res1), torch.cat(y_runtime_res2))
        self.assertEqual(y_ref, y_runtime_res)

        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, 0))

        multi_stream_model2 = self.create_multi_stream_module(traced_model,
                                                                    num_streams,
                                                                    cpu_pool,
                                                                    multi_stream_input_hint,
                                                                    multi_stream_output_hint,
                                                                    concat_output=True)
        y_runtime_res2 = multi_stream_model2(key1=input_tensor1, key2=input_tensor2, key3=input_tensor3)

        self.assertEqual(y_ref, y_runtime_res2)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_mix_position_keyword_input_output_hint(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, param1, param2, key1 = None):
                return param1, param2, key1

        batch_size, num_streams, cpu_pool = self.init_set_up()
        # This module simulates the behaviour of Bert Large LZ models:
        #   * Accept 3 tensors (2 position parameter and 1 key word parameter) as input
        #   * Return a tuple of 3 tensors as output
        model = TestModule().eval()

        jit_input = (torch.rand(batch_size, 1),
                torch.rand(batch_size, 2),
                torch.rand(batch_size, 3))

        traced_model = self.create_jit_traced_model(model, jit_input)

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 2)
        input_tensor3 = torch.rand(batch_size, 3)
        input = (input_tensor1, input_tensor2)
        k_input = {"key1":input_tensor3}

        # Warm Up
        for _ in range(3):
            traced_model(input_tensor1, input_tensor2, key1=input_tensor3)

        # Calculate the reference result
        y_ref = traced_model(*input, **k_input)
        y_ref2 = traced_model(input_tensor1, input_tensor2, input_tensor3)
        y_ref3 = traced_model(input_tensor1, input_tensor2, key1 = input_tensor3)
        self.assertEqual(y_ref, y_ref2)
        self.assertEqual(y_ref, y_ref3)

        # Be careful, jit traced model will change the input type
        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(0, 0, key1=0)

        # Create Multi Stream Module with concat output
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, 0, 0))

        multi_stream_model = self.create_multi_stream_module(traced_model,
                                                            num_streams,
                                                            cpu_pool,
                                                            multi_stream_input_hint,
                                                            multi_stream_output_hint,
                                                            concat_output=True)
        # There are 2 ways to write now
        y_runtime_res = multi_stream_model(input_tensor1, input_tensor2, key1 = input_tensor3)
        y_runtime_res2 = multi_stream_model(*input, **k_input)
        self.assertEqual(y_ref, y_runtime_res)
        self.assertEqual(y_ref, y_runtime_res2)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_input_output_hint_not_along_dim_zero(self):
        batch_size, num_streams, cpu_pool = self.init_set_up()

        # This module:
        #   * Accept 3 tensors as input
        #   * Return a tuple of 3 tensors as output
        model = TestInputOutputModule().eval()

        input_tensor1 = torch.rand(1, batch_size)
        input_tensor2 = torch.rand(batch_size, 2)
        input_tensor3 = torch.rand(3, batch_size)

        # Since jit trace only accept single tensor or a tuple of tensors as input
        # https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace
        jit_input = (input_tensor1, input_tensor2, input_tensor3)

        traced_model = self.create_jit_traced_model(model, jit_input)

        # Warm Up in the main thread to finish the jit pass optimizations
        for _ in range(3):
            traced_model(input_tensor1, input_tensor2, input_tensor3)

        # Calculate the reference result
        y_ref = traced_model(input_tensor1, input_tensor2, input_tensor3)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(1, 0, 1)

        multi_stream_model = self.create_multi_stream_module(traced_model,
                                                            num_streams,
                                                            cpu_pool,
                                                            multi_stream_input_hint,
                                                            concat_output=False)
        y_runtime = multi_stream_model(input_tensor1, input_tensor2, input_tensor3)

        # Manually concat the output
        y_runtime_res1 = []
        y_runtime_res2 = []
        y_runtime_res3 = []
        for stream_id in range(num_streams if ((batch_size // num_streams) >= 1) else batch_size):
            y_runtime_res1.append(y_runtime[stream_id][0])
            y_runtime_res2.append(y_runtime[stream_id][1])
            y_runtime_res3.append(y_runtime[stream_id][2])
        y_runtime_res = (torch.cat(y_runtime_res1, 1), torch.cat(y_runtime_res2, 0), torch.cat(y_runtime_res3, 1))
        self.assertEqual(y_ref, y_runtime_res)

        # Create Multi Stream Module with concat output
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((1, 0, 1))

        multi_stream_model2 = self.create_multi_stream_module(traced_model,
                                                                    num_streams,
                                                                    cpu_pool,
                                                                    multi_stream_input_hint,
                                                                    multi_stream_output_hint,
                                                                    concat_output=True)
        y_runtime_res2 = multi_stream_model2(input_tensor1, input_tensor2, input_tensor3)
        self.assertEqual(y_ref, y_runtime_res2)

class TestMultiStreamBenchmarkModule(JitTestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    def test_multi_stream_benchmark_module_bf16_jit_model(self):
        model = SimpleNet().eval()
        batch_size = 1
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
            trace_model = torch.jit.trace(model, x)
        # Warm Up
        for _ in range(3):
            trace_model(x)

        # Create MultiStreamModule
        multi_stream_model = ipex.cpu.runtime._MultiStreamBenchmarkModule(trace_model)
        multi_stream_model(x)

if __name__ == '__main__':
    test = unittest.main()
