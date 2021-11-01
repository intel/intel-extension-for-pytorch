import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from .cpupool import CPUPool

class MultiStreamModule(nn.Module):
    r"""MultiStreamModule supports multi stream throughput running method.
        Args: `model` the input model.
              `num_streams` how instances we want to run.
              `cpu_pool` CPUPool type includes all the cores used to run multi stream inference.
    """
    def __init__(self, model, num_streams: int, cpu_pool: CPUPool):
        super(MultiStreamModule, self).__init__()
        assert type(cpu_pool) is CPUPool
        core_list = cpu_pool.core_ids
        self.num_streams = num_streams
        self.cores_per_instance = core_list.__len__() // self.num_streams
        assert core_list.__len__() % self.cores_per_instance == 0, "the cores number:{0} in node:{1} are not divisible by cores_per_instance:{2}".format(core_list.__len__(), node_id, cores_per_instance)
        self.tasks = []
        for j in range(self.num_streams):
            start_core_id = core_list[j*self.cores_per_instance]
            end_core_id = start_core_id + self.cores_per_instance
            self.tasks.append(ipex.cpu.runtime.Task(model, ipex.cpu.runtime.CPUPool(range(start_core_id, end_core_id))))

    def forward(self, inputs):
        # Ensure each instance has input offload
        batch_per_instance = inputs.size(0) // self.num_streams
        if batch_per_instance >= 1:
            # the input size large or equal number of instance
            used_num_streams = self.num_streams
            instance_need_extra_input = inputs.size(0) % self.num_streams # input image size large than num_streams and not divisible
        else:
            # the input size less than number of instance
            batch_per_instance = 1
            used_num_streams = inputs.size(0)
            instance_need_extra_input = 0
        results_raw_future = []
        results_raw = []
        start_idx = 0
        end_idx = 0
        for j in range(used_num_streams):
            if j < instance_need_extra_input:
                # tail case, when the input image size large than num_streams and not divisible
                end_idx = end_idx + (batch_per_instance + 1)
            else:
                # input image size divisible of num_streams or input image size less than num_streams
                end_idx = end_idx + batch_per_instance
            results_raw_future.append(self.tasks[j](inputs[start_idx:end_idx]))
            start_idx = end_idx

        for j in range(used_num_streams):
            results_raw.append(results_raw_future[j].get())
        output = torch.cat(results_raw)
        return output
