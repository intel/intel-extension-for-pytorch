import unittest
import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

from common_ipex_conf import runtime_thread_affinity_test_env
import subprocess
import os


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(
            64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y


class SimpleNet_v2(torch.nn.Module):
    def __init__(self):
        super(SimpleNet_v2, self).__init__()
        self.conv = torch.nn.Conv2d(
            3, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            64, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.conv2(x1)
        y = torch.flatten(x1, start_dim=1)
        return y


class SimpleNet_dict(torch.nn.Module):
    def __init__(self):
        super(SimpleNet_dict, self).__init__()
        self.conv = torch.nn.Conv2d(
            64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, **x_dict):
        x1 = self.conv(x_dict["x1"])
        x2 = self.conv(x_dict["x2"])
        y1 = x1 + x2
        y2 = torch.flatten(y1, start_dim=1)
        ret_dict = {"y1": y1, "y2": y2}
        return ret_dict


class SimpleNet_tensor_dict(torch.nn.Module):
    def __init__(self):
        super(SimpleNet_tensor_dict, self).__init__()
        self.conv = torch.nn.Conv2d(
            64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, **x_dict):
        x1 = self.conv(x_dict["x1"])
        x2 = self.conv(x_dict["x2"])
        y1 = x1 + x2
        y2 = torch.flatten(y1, start_dim=1)
        ret_dict = {"y1": y1, "y2": y2}
        # Return a tuple of (Tensor, dict)
        return y1, ret_dict


class TestInputOutputModule(torch.nn.Module):
    def __init__(self):
        super(TestInputOutputModule, self).__init__()

    def forward(self, *args, **kwargs):
        return args


class TestInputOutputModule2(torch.nn.Module):
    def __init__(self):
        super(TestInputOutputModule2, self).__init__()

    def forward(self, param1):
        return param1


class TestCPUPool(TestCase):
    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    def test_cpupool_get_core_list(self):
        core_list = [0, 1]
        cpu_pool = ipex.cpu.runtime.CPUPool(core_list)
        self.assertEqual(cpu_pool.cpu_pool.get_core_list(), core_list)


class TestCoreBinding(TestCase):
    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_decorator_imperative_model(self):
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

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_with_context_imperative_model(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)
        cpu_pool = ipex.cpu.runtime.CPUPool([1, 2, 3, 4])
        with ipex.cpu.runtime.pin(cpu_pool):
            y_runtime = model(x)
        y = model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_nested_with_context_imperative_model(self):
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
    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_task_async_api_imperative_model(self):
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

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_task_sync_api_imperative_model(self):
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

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_task_async_api_native_function(self):
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

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_task_copy(self):
        model = SimpleNet()
        model.eval()
        x = torch.rand(64, 64, 3, 3)
        # Calculate the reference result
        y = model(x)

        # Create task
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        task = ipex.cpu.runtime.Task(model, cpu_pool)

        # Copy task
        task2 = task

        # Task submit and wait
        y_runtime_future = task(x)
        y_runtime = y_runtime_future.get()
        y_runtime_future2 = task2(x)
        y_runtime2 = y_runtime_future2.get()
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, y_runtime2)


class TestMultiStreamModule(TestCase):
    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_multi_stream_module(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=2, cpu_pool=cpu_pool
        )

        y_runtime = multi_stream_model(x)
        self.assertEqual(y, y_runtime)

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_multi_stream_module_with_dict_return_type(self):
        model = SimpleNet_dict()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x1 = torch.rand(batch_size, 64, 3, 3)
        x2 = torch.rand(batch_size, 64, 3, 3)
        x_dict = {"x1": x1, "x2": x2}

        # Calculate the reference result
        y_dict = model(**x_dict)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)

        input_hint_object = {"x1": 0, "x2": 0}
        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            **input_hint_object
        )
        output_concat_object = {"y1": 0, "y2": 0}
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            **output_concat_object
        )

        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model,
            num_streams=2,
            cpu_pool=cpu_pool,
            input_split_hint=multi_stream_input_hint,
            output_concat_hint=multi_stream_output_hint,
        )

        y_runtime_dict = multi_stream_model(**x_dict)
        self.assertEqual(y_dict["y1"], y_runtime_dict["y1"])
        self.assertEqual(y_dict["y2"], y_runtime_dict["y2"])

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_multi_stream_module_with_tensor_and_dict_return_type(self):
        model = SimpleNet_tensor_dict()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x1 = torch.rand(batch_size, 64, 3, 3)
        x2 = torch.rand(batch_size, 64, 3, 3)
        x_dict = {"x1": x1, "x2": x2}

        # Calculate the reference result
        y, y_dict = model(**x_dict)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)

        input_hint_object = {"x1": 0, "x2": 0}
        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            **input_hint_object
        )
        output_concat_object = (0, {"y1": 0, "y2": 0})
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            output_concat_object
        )

        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model,
            num_streams=2,
            cpu_pool=cpu_pool,
            input_split_hint=multi_stream_input_hint,
            output_concat_hint=multi_stream_output_hint,
        )

        y_runtime, y_runtime_dict = multi_stream_model(**x_dict)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y_dict["y1"], y_runtime_dict["y1"])
        self.assertEqual(y_dict["y2"], y_runtime_dict["y2"])

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_single_stream_module(self):
        model = SimpleNet()
        model.eval()
        batch_size = ipex.cpu.runtime.get_core_list_of_node_id(0).__len__()
        x = torch.rand(batch_size, 64, 3, 3)

        # Calculate the reference result
        y = model(x)

        # Create MultiStreamModule
        cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=1, cpu_pool=cpu_pool
        )
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=1, cpu_pool=cpu_pool, concat_output=False
        )

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, y_runtime2[0])

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_core_number_not_divisible_by_stream_number(self):
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
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool
        )
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False
        )

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
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
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool
        )
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False
        )

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_batchsize_not_divisible_by_stream_number(self):
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
        multi_stream_model = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool
        )
        multi_stream_model2 = ipex.cpu.runtime.MultiStreamModule(
            model, num_streams=num_streams, cpu_pool=cpu_pool, concat_output=False
        )

        y_runtime = multi_stream_model(x)
        y_runtime2 = multi_stream_model2(x)
        self.assertEqual(y, y_runtime)
        self.assertEqual(y, torch.cat(y_runtime2))
        self.assertEqual(y_runtime2[0].size(0), 2)
        self.assertEqual(y_runtime2[1].size(0), 1)
        self.assertEqual(y_runtime2[2].size(0), 1)


class TestModuleMultiStreamModuleHint(TestCase):
    # For the inputs format which can't be jit.trace
    def init_set_up(self):
        # Create Multi Stream Module without concat output
        cpu_pool = ipex.cpu.runtime.CPUPool()
        batch_size = cpu_pool.core_ids.__len__()
        num_streams = cpu_pool.core_ids.__len__()
        return batch_size, num_streams, cpu_pool

    def create_multi_stream_module(
        self,
        traced_model,
        num_streams,
        cpu_pool,
        multi_stream_input_hint,
        multi_stream_output_hint=None,
        concat_output=True,
    ):
        if not concat_output:
            return ipex.cpu.runtime.MultiStreamModule(
                traced_model,
                num_streams=num_streams,
                cpu_pool=cpu_pool,
                concat_output=False,
                input_split_hint=multi_stream_input_hint,
            )
        else:
            return ipex.cpu.runtime.MultiStreamModule(
                traced_model,
                num_streams=num_streams,
                cpu_pool=cpu_pool,
                input_split_hint=multi_stream_input_hint,
                output_concat_hint=multi_stream_output_hint,
            )

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_mix_tensor_bool_input_output_hint(self):
        # This module:
        #   * Accept 2 tensors + 1 scalar as input
        #   * Return 2 tensors + 1 scalar as output
        # Since Type 'Tuple[Tensor, bool, Tensor]' cannot be traced, we put this test input type in imperative mode.
        model = TestInputOutputModule().eval()
        batch_size, num_streams, cpu_pool = self.init_set_up()

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 3)

        # Calculate the reference result
        y_ref = model(input_tensor1, False, input_tensor2)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(0, None, 0)
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, None, 0))

        multi_stream_model = self.create_multi_stream_module(
            model,
            num_streams,
            cpu_pool,
            multi_stream_input_hint,
            multi_stream_output_hint,
            concat_output=True,
        )
        y_runtime_res = multi_stream_model(input_tensor1, False, input_tensor2)
        self.assertEqual(y_ref, y_runtime_res)

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_tuple_input_output_hint(self):
        # This module:
        #   * Accept 1 tuple(3 tensors) as input
        #   * Return 1 tuple(3 tensors) as output
        model = TestInputOutputModule2().eval()
        batch_size, num_streams, cpu_pool = self.init_set_up()

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 2)
        input_tensor3 = torch.rand(batch_size, 3)
        input = (input_tensor1, input_tensor2, input_tensor3)
        y_ref = model(input)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, 0, 0))
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint((0, 0, 0))

        multi_stream_model = self.create_multi_stream_module(
            model,
            num_streams,
            cpu_pool,
            multi_stream_input_hint,
            multi_stream_output_hint,
            concat_output=True,
        )
        y_runtime_res = multi_stream_model(input)
        self.assertEqual(y_ref, y_runtime_res)

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_dict_input_output_hint(self):
        # This module:
        #   * Accept 1 dict(3 tensors) as input
        #   * Return 1 dict(3 tensors) as output
        model = TestInputOutputModule2().eval()
        batch_size, num_streams, cpu_pool = self.init_set_up()

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 2)
        input_tensor3 = torch.rand(batch_size, 3)
        input = {"key1": input_tensor1, "key2": input_tensor2, "key3": input_tensor3}
        y_ref = model(input)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            {"key1": 0, "key2": 0, "key3": 0}
        )
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint(
            {"key1": 0, "key2": 0, "key3": 0}
        )

        multi_stream_model = self.create_multi_stream_module(
            model,
            num_streams,
            cpu_pool,
            multi_stream_input_hint,
            multi_stream_output_hint,
            concat_output=True,
        )
        y_runtime_res = multi_stream_model(input)
        self.assertEqual(y_ref, y_runtime_res)

    @unittest.skipIf(
        not ipex.cpu.runtime.is_runtime_ext_enabled(),
        "Skip when IPEX Runtime extension is not enabled",
    )
    @runtime_thread_affinity_test_env
    def test_nested_tuple_input_output_hint(self):
        # This module:
        #   * Accept nested tuple ((tensor1, tensor2), tensor3) as input
        #   * Return nested tuple ((tensor1, tensor2), tensor3) as output
        model = TestInputOutputModule2().eval()
        batch_size, num_streams, cpu_pool = self.init_set_up()

        input_tensor1 = torch.rand(batch_size, 1)
        input_tensor2 = torch.rand(batch_size, 2)
        input_tensor3 = torch.rand(batch_size, 3)
        input = ((input_tensor1, input_tensor2), input_tensor3)
        y_ref = model(input)

        multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(((0, 0), 0))
        multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint(((0, 0), 0))

        multi_stream_model = self.create_multi_stream_module(
            model,
            num_streams,
            cpu_pool,
            multi_stream_input_hint,
            multi_stream_output_hint,
            concat_output=True,
        )
        y_runtime_res = multi_stream_model(input)
        self.assertEqual(y_ref, y_runtime_res)


def is_numactl_available():
    numactl_available = False
    cmd = ["numactl", "-C", "0", "-m", "0", "ls"]
    try:
        r = subprocess.run(cmd, env=os.environ)
    except BaseException:
        return numactl_available
    if r.returncode == 0:
        numactl_available = True
    return numactl_available


class TestRuntimeExtensionWithNumactl(TestCase):
    @unittest.skipIf(
        not (is_numactl_available() and ipex.cpu.runtime.is_runtime_ext_enabled()),
        "Skip when numactl is not available",
    )
    @runtime_thread_affinity_test_env
    def test_cpupool_creation_with_numactl(self):
        loc = os.path.dirname(os.path.abspath(__file__))
        cmd1 = "numactl -C 0-1 -m 0 python -u {}/runtime.py --case-name={}".format(
            loc, "create_cpu_pool"
        )
        cmd2 = "OMP_NUM_THREADS=1 KMP_AFFINITY=granularity=fine,compact,1,0 numactl -C 0-1 -m 0 \
            python -u {}/runtime.py --case-name={}".format(
            loc, "create_cpu_pool"
        )
        cmds = [cmd1, cmd2]
        for cmd in cmds:
            match = False
            with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as p:
                for line in p.stdout.readlines():
                    line = str(line, "utf-8").strip()
                    if "The created CPUPool has core is:" in line:
                        x = line.split(":")
                        assert (
                            "[1]" in x[1]
                        ), "The core ids in test_cpupool_creation with numactl is not as expected."
                        match = True
            assert match, "Test Case Failed to create CPUPool"


if __name__ == "__main__":
    test = unittest.main()
