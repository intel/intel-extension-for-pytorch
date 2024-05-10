import torch
import intel_extension_for_pytorch._C as torch_ipex_cpp


def has_ccl():
    return hasattr(torch.ops.torch_ipex, "all_reduce_add")


if has_ccl():
    get_world_size = torch_ipex_cpp.get_world_size
    get_rank = torch_ipex_cpp.get_rank
    barrier = torch_ipex_cpp.barrier
    allreduce_add = torch.ops.torch_ipex.all_reduce_add
    allgather = torch.ops.torch_ipex.allgather
