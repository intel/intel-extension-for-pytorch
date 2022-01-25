import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from .cpupool import CPUPool

class MultiStreamModule(nn.Module):
    r"""
    MultiStreamModule supports inference with multi-stream throughput mode.

    If the number of cores inside ``cpu_pool`` is divisible by ``num_streams``,
    the cores will be allocated equally to each stream.

    If the number of cores inside ``cpu_pool`` is not divisible by
    ``num_streams`` with remainder N, one extra core will be allocated to the
    first N streams.

    Args:
        model (torch.jit.ScriptModule or torch.nn.Module): The input model.
        num_streams (int): Number of instances.
        cpu_pool (intel_extension_for_pytorch.cpu.runtime.CPUPool): An
            intel_extension_for_pytorch.cpu.runtime.CPUPool object, contains
            all CPU cores used to run multi-stream inference.
        concat_output (bool): A flag indicates whether the output of each
            stream will be concatenated or not. The default value is True. Note:
            if the output of each stream can't be concatenated, set this flag to
            false to get the raw output (a list of each stream's output).

    Returns:
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModule: Generated
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModule object.

    :meta public:
    """

    def __init__(self, model, num_streams: int, cpu_pool: CPUPool, concat_output: bool = True):
        super(MultiStreamModule, self).__init__()
        assert type(cpu_pool) is CPUPool
        self.core_list = cpu_pool.core_ids
        self.num_streams = num_streams
        if self.num_streams == 1:
            # Sync execution path if num_stream is 1.
            self.model = model
        else:
            self.cores_per_instance = self.core_list.__len__() // self.num_streams
            num_stream_allocated_extra_core = self.core_list.__len__() % self.num_streams
            self.tasks = []
            start_core_list_idx = 0
            end_core_list_idx = 0
            for j in range(self.num_streams):
                if j < num_stream_allocated_extra_core:
                    # If the core number is not divisible by stream number,
                    # the remainder streams(num_stream_allocated_extra_core) will be allocated one extra core.
                    end_core_list_idx += (self.cores_per_instance + 1)
                else:
                    end_core_list_idx += self.cores_per_instance
                self.tasks.append(ipex.cpu.runtime.Task(model, ipex.cpu.runtime.CPUPool(self.core_list[start_core_list_idx:end_core_list_idx])))
                start_core_list_idx = end_core_list_idx
        self.concat_output = concat_output

    def forward(self, inputs):
        if self.num_streams == 1:
            # Sync execution path if num_stream is 1
            if not ipex._C.is_same_core_affinity_setting(self.core_list):
                # If the main thread's core affinity has been changed, we should set it again.
                ipex._C.pin_cpu_cores(self.core_list)
            results_raw = self.model(inputs)
            return results_raw if self.concat_output else [results_raw]
        # Ensure each instance has input offload
        batch_per_instance = inputs.size(0) // self.num_streams
        if batch_per_instance >= 1:
            # The input batchsize larger or equal to num_streams.
            used_num_streams = self.num_streams
            # If input batchsize larger than num_streams and not divisible,
            # the first remainder streams will have (mini_batch + 1) input size.
            instance_need_extra_input = inputs.size(0) % self.num_streams
        else:
            # The input batchsize less than num_streams,
            # only the first batchsize stream will have mini_batch(1) input.
            batch_per_instance = 1
            used_num_streams = inputs.size(0)
            instance_need_extra_input = 0
        results_raw_future = []
        results_raw = []
        start_idx = 0
        end_idx = 0
        for j in range(used_num_streams):
            if j < instance_need_extra_input:
                # Tail case, when the input image size larger than num_streams and not divisible,
                # the first remainder streams will have (mini_batch + 1) input size.
                end_idx = end_idx + (batch_per_instance + 1)
            else:
                # Input image size divisible of num_streams or input image size less than num_streams.
                end_idx = end_idx + batch_per_instance
            results_raw_future.append(self.tasks[j](inputs[start_idx:end_idx]))
            start_idx = end_idx

        for j in range(used_num_streams):
            results_raw.append(results_raw_future[j].get())
        return torch.cat(results_raw) if self.concat_output else results_raw
