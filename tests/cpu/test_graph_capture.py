import unittest
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
from common_ipex_conf import runtime_thread_affinity_test_env
from torch.utils import ThroughputBenchmark

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

class Conv_Bn_Relu(nn.Module):
    def __init__(self):
        super(Conv_Bn_Relu, self).__init__()

        self.conv = torch.nn.Conv2d(6, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class Conv_IF_Relu(nn.Module):
    def __init__(self):
        super(Conv_IF_Relu, self).__init__()

        self.conv = torch.nn.Conv2d(6, 3, 3)

    def forward(self, x):
        if x.sum().item() > 0:
            return F.relu(self.conv(x), inplace=True)
        else:
            return F.relu(self.conv(x))

class TestGraphCapture(TestCase):

    def test_inference_graph_mode_jit(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        model = ipex.optimize(model, graph_mode=True)

        with torch.no_grad():
            for _ in range(10):
                y2 = model(x)
        self.assertEqual(y1, y2)

    def test_inference_graph_mode_torchdynamo(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        model = ipex.optimize(model, graph_mode=True)

        with torch.no_grad():
            for _ in range(10):
                y2 = model(x)
        self.assertEqual(y1, y2)

    def test_inference_graph_mode_jit_autocast(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        with torch.cpu.amp.autocast(), torch.no_grad():
            for _ in range(10):
                y2_bf16 = model(x)
        self.assertEqual(y1, y2_bf16, prec=0.01)
        self.assertTrue(y2_bf16.dtype == torch.bfloat16)

    def test_inference_graph_mode_torchdynamo_autocast(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        with torch.cpu.amp.autocast(), torch.no_grad():
            for _ in range(10):
                y2_bf16 = model(x)
        self.assertEqual(y1, y2_bf16, prec=0.01)
        self.assertTrue(y2_bf16.dtype == torch.bfloat16)

    def test_inference_trace_graph_mode(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        # JIT trace and freeze
        traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)
        # graph capture
        traced_model = ipex.optimize(traced_model, graph_mode=True)

        with torch.no_grad():
            for _ in range(10):
                y2 = traced_model(x)
        self.assertEqual(y1, y2)

        freeze_graph = traced_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(traced_model, torch.jit.RecursiveScriptModule))

        # JIT save, load
        torch.jit.save(traced_model, 'scriptmodule.pt')
        load_model = torch.jit.load('scriptmodule.pt')

        with torch.no_grad():
            for _ in range(10):
                y3 = load_model(x)
        self.assertEqual(y1, y3)

        freeze_graph = load_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(load_model, torch.jit.RecursiveScriptModule))
        os.remove('scriptmodule.pt')

    def test_inference_graph_mode_trace(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        # graph capture
        model = ipex.optimize(model, graph_mode=True)

        # JIT trace and freeze
        traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)

        with torch.no_grad():
            for _ in range(10):
                y2 = traced_model(x)
        self.assertEqual(y1, y2)

        freeze_graph = traced_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(traced_model, torch.jit.RecursiveScriptModule))

        # JIT save, load
        torch.jit.save(traced_model, 'scriptmodule.pt')
        load_model = torch.jit.load('scriptmodule.pt')

        with torch.no_grad():
            for _ in range(10):
                y3 = load_model(x)
        self.assertEqual(y1, y3)

        freeze_graph = load_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(load_model, torch.jit.RecursiveScriptModule))
        os.remove('scriptmodule.pt')

    def test_inference_graph_mode_trace2(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        y1 = model(x)
        # graph capture
        model = ipex.optimize(model, graph_mode=True)

        with torch.no_grad():
            for _ in range(10):
                y2 = model(x)
        self.assertEqual(y1, y2)

        # JIT trace and freeze
        traced_model = torch.jit.trace(model, x)
        traced_model = torch.jit.freeze(traced_model)

        with torch.no_grad():
            for _ in range(10):
                y3 = traced_model(x)
        self.assertEqual(y1, y3)

        freeze_graph = traced_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(traced_model, torch.jit.RecursiveScriptModule))

        # JIT save, load
        torch.jit.save(traced_model, 'scriptmodule.pt')
        load_model = torch.jit.load('scriptmodule.pt')

        with torch.no_grad():
            for _ in range(10):
                y4 = load_model(x)
        self.assertEqual(y1, y4)

        freeze_graph = load_model.graph_for(x)
        self.assertTrue(any(n.kind() == "ipex_prepack::convolution_relu_run" for n in freeze_graph.nodes()))
        self.assertTrue(isinstance(load_model, torch.jit.RecursiveScriptModule))
        os.remove('scriptmodule.pt')

    def test_throughput_benchmark_graph_mode_jit(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(3, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, graph_mode=True)

        bench = ThroughputBenchmark(model)
        bench.add_input(x)
        bench.benchmark(
                num_calling_threads=14,
                num_warmup_iters=10,
                num_iters=100)

        y_bench = bench.run_once(x)

        # Calculate the reference result
        y = model(x)
        self.assertEqual(y, y_bench)

    def test_throughput_benchmark_graph_mode_torchdynamo(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(3, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, graph_mode=True)

        bench = ThroughputBenchmark(model)
        bench.add_input(x)
        bench.benchmark(
                num_calling_threads=14,
                num_warmup_iters=10,
                num_iters=100)

        y_bench = bench.run_once(x)

        # Calculate the reference result
        y = model(x)
        self.assertEqual(y, y_bench)

    def test_throughput_benchmark_graph_mode_jit_autocast(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(3, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        bench = ThroughputBenchmark(model)
        bench.add_input(x)
        with torch.cpu.amp.autocast():
            bench.benchmark(
                    num_calling_threads=14,
                    num_warmup_iters=10,
                    num_iters=100)

            y_bench = bench.run_once(x)

            # Calculate the reference result
            y = model(x)
        self.assertEqual(y, y_bench)
        self.assertTrue(y_bench.dtype == torch.bfloat16)

    def test_throughput_benchmark_graph_mode_torchdynamo_autocast(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(3, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        bench = ThroughputBenchmark(model)
        bench.add_input(x)
        with torch.cpu.amp.autocast():
            bench.benchmark(
                    num_calling_threads=14,
                    num_warmup_iters=10,
                    num_iters=100)

            y_bench = bench.run_once(x)

            # Calculate the reference result
            y = model(x)
        self.assertEqual(y, y_bench)
        self.assertTrue(y_bench.dtype == torch.bfloat16)

    @skipIfNoTorchVision
    def test_resnet50(self):
        model = torchvision.models.resnet50(pretrained=False)
        model.eval()
        data = torch.rand(2, 3, 224, 224)
        model = model.to(memory_format=torch.channels_last)
        data = data.to(memory_format=torch.channels_last)

        model = ipex.optimize(model, graph_mode=True)

        with torch.no_grad():
            for _ in range(10):
                y = model(data)
        self.assertTrue(y.dtype == torch.float32)

    @skipIfNoTorchVision
    def test_resnet50_autocast(self):
        model = torchvision.models.resnet50(pretrained=False)
        model.eval()
        data = torch.rand(2, 3, 224, 224)
        model = model.to(memory_format=torch.channels_last)
        data = data.to(memory_format=torch.channels_last)

        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        with torch.cpu.amp.autocast(), torch.no_grad():
            for _ in range(10):
                y = model(data)
        self.assertTrue(y.dtype == torch.bfloat16)

    def test_training_graph_mode_jit(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).train()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        x1 = copy.deepcopy(x).requires_grad_()
        x2 = copy.deepcopy(x).requires_grad_()
        y1 = model(x1)
        y1.sum().backward()

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        model, opt = ipex.optimize(model, optimizer=sgd, graph_mode=True)
        y2 = model(x2)
        y2.sum().backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)

    def test_training_graph_mode_fallback(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last).train()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        x1 = copy.deepcopy(x).requires_grad_()
        x2 = copy.deepcopy(x).requires_grad_()
        y1 = model(x1)
        y1.sum().backward()

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        model, opt = ipex.optimize(model, optimizer=sgd, graph_mode=True)
        y2 = model(x2)
        y2.sum().backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)

    def test_training_graph_mode_jit_autocast(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).train()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        x1 = copy.deepcopy(x).requires_grad_()
        x2 = copy.deepcopy(x).requires_grad_()
        y1 = model(x1)
        y1.sum().backward()

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        model, opt = ipex.optimize(model, optimizer=sgd, dtype=torch.bfloat16, graph_mode=True)
        with torch.cpu.amp.autocast():
            y2 = model(x2)
            y2.sum().backward()

        self.assertEqual(y1, y2, prec=0.1)
        self.assertEqual(x1.grad, x2.grad, prec=0.01)
        self.assertTrue(y2.dtype == torch.bfloat16)

    def test_training_graph_mode_fallback_autocast(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last).train()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        x1 = copy.deepcopy(x).requires_grad_()
        x2 = copy.deepcopy(x).requires_grad_()
        y1 = model(x1)
        y1.sum().backward()

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        model, opt = ipex.optimize(model, optimizer=sgd, dtype=torch.bfloat16, graph_mode=True)
        with torch.cpu.amp.autocast():
            y2 = model(x2)
            y2.sum().backward()

        self.assertEqual(y1, y2, prec=0.1)
        self.assertEqual(x1.grad, x2.grad, prec=0.01)
        self.assertTrue(y2.dtype == torch.bfloat16)

    def test_training_save_load(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last).train()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        origin_x = x.clone()
        ipex_x = x.clone()
        origin_model = copy.deepcopy(model).train()
        lr = 1e-2
        origin_optimizer = torch.optim.SGD(origin_model.parameters(), lr=lr)
        ipex_model, ipex_optimizer = ipex.optimize(origin_model, optimizer=origin_optimizer, graph_mode=True)
        # train one step for origin.
        y1 = origin_model(origin_x)
        loss1 = y1.sum()
        origin_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_value_(origin_model.parameters(), 10)
        origin_optimizer.step()
        # train one step for ipex.
        y2 = ipex_model(ipex_x)
        loss2 = y2.sum()
        ipex_optimizer.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_value_(ipex_model.parameters(), 10)
        ipex_optimizer.step()
        torch.save({'model_state_dict': origin_model.state_dict(),
                    'optimizer_state_dict': origin_optimizer.state_dict()
                    }, 'origin_checkpoint.pth')
        torch.save({'model_state_dict': ipex_model.state_dict(),
                    'optimizer_state_dict': ipex_optimizer.state_dict()
                    }, 'ipex_checkpoint.pth')
        self.assertEqual(y1, y2)
        origin_model_state = origin_model.state_dict()
        ipex_model_state = ipex_model.state_dict()
        for var_name in origin_model_state:
            self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
        # check state_buffer works.
        origin_optimizer_state = origin_optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in origin_optimizer_state:
            if var_name == 'state':
                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name])

        origin_model = copy.deepcopy(model).train()
        origin_optimizer = torch.optim.SGD(origin_model.parameters(), lr=lr)
        origin_checkpoint = torch.load('origin_checkpoint.pth')
        origin_model.load_state_dict(origin_checkpoint['model_state_dict'])
        origin_optimizer.load_state_dict(origin_checkpoint['optimizer_state_dict'])
        # load ipex model state
        origin_ipex_model = copy.deepcopy(model)
        origin_ipex_optimizer = torch.optim.SGD(origin_ipex_model.parameters(), lr=lr)
        ipex_checkpoint = torch.load('ipex_checkpoint.pth')
        origin_ipex_model.load_state_dict(ipex_checkpoint['model_state_dict'])
        origin_ipex_optimizer.load_state_dict(ipex_checkpoint['optimizer_state_dict'])
        ipex_model, ipex_optimizer = ipex.optimize(origin_model, optimizer=origin_optimizer, graph_mode=True)
        # train second step for origin.
        y1 = origin_model(origin_x)
        loss = y1.sum()
        origin_optimizer.zero_grad()
        loss.backward()
        origin_optimizer.step()
        # traing second step for ipex model.
        y3 = ipex_model(ipex_x)
        loss3 = y3.sum()
        ipex_optimizer.zero_grad()
        loss3.backward()
        ipex_optimizer.step()
        self.assertEqual(y1, y3)
        origin_model_state = origin_model.state_dict()
        ipex_model_state = ipex_model.state_dict()
        for var_name in origin_model_state:
            self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
        # check state_buffer works.
        origin_optimizer_state = origin_optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in origin_optimizer_state:
            if var_name == 'state':
                self.assertEqual(origin_optimizer_state[var_name], ipex_optimizer_state[var_name])
        os.remove('origin_checkpoint.pth')
        os.remove('ipex_checkpoint.pth')

class TestGraphCaptureMultiStream(TestCase):
    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    @runtime_thread_affinity_test_env
    def test_multi_stream_graph_mode_jit(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(56, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, graph_mode=True)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=28, cpu_pool=cpu_pool)

        for _ in range(10):
            y_runtime = multi_stream_model(x)

        # Calculate the reference result
        y = model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    @runtime_thread_affinity_test_env
    def test_multi_stream_graph_mode_torchdynamo(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(56, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, graph_mode=True)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=28, cpu_pool=cpu_pool)

        for _ in range(10):
            y_runtime = multi_stream_model(x)

        # Calculate the reference result
        y = model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    @runtime_thread_affinity_test_env
    def test_multi_stream_graph_mode_jit_autocast(self):
        model = Conv_Bn_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(56, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=28, cpu_pool=cpu_pool)

        with torch.cpu.amp.autocast():
            for _ in range(10):
                y_runtime = multi_stream_model(x)

            # Calculate the reference result
            y = model(x)
        self.assertEqual(y, y_runtime)
        self.assertTrue(y_runtime.dtype == torch.bfloat16)

    @unittest.skipIf(not ipex.cpu.runtime.is_runtime_ext_enabled(), "Skip when IPEX Runtime extension is not enabled")
    @runtime_thread_affinity_test_env
    def test_multi_stream_graph_mode_torchdynamo_autocast(self):
        model = Conv_IF_Relu().to(memory_format=torch.channels_last)
        model.eval()
        x = torch.rand(56, 6, 10, 10).to(memory_format=torch.channels_last)

        model = ipex.optimize(model, dtype=torch.bfloat16, graph_mode=True)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=28, cpu_pool=cpu_pool)

        with torch.cpu.amp.autocast():
            for _ in range(10):
                y_runtime = multi_stream_model(x)

            # Calculate the reference result
            y = model(x)
        self.assertEqual(y, y_runtime)
        self.assertTrue(y_runtime.dtype == torch.bfloat16)


if __name__ == '__main__':
    test = unittest.main()
