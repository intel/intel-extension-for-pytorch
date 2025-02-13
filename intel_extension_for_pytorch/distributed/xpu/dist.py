import torch
import torch.distributed as dist
from typing import List


def all_reduce_xpu(t: torch.Tensor, op, group=None, async_op=False):
    try:
        import oneccl_bindings_for_pytorch  # noqa
    except ImportError as e:
        raise RuntimeError("oneccl_bindings_for_pytorch is not installed!") from e
    if not dist.is_initialized():
        dist.init_process_group("ccl")
    return dist.all_reduce(t, op, group, async_op)


def all_gather_xpu(
    t_list: List[torch.Tensor], t: torch.Tensor, group=None, async_op=False
):
    try:
        import oneccl_bindings_for_pytorch  # noqa
    except ImportError as e:
        raise RuntimeError("oneccl_bindings_for_pytorch is not installed!") from e
    if not dist.is_initialized():
        dist.init_process_group("ccl")
    return dist.all_gather(t_list, t, group, async_op)


def all_gather_into_tensor_xpu(output_tensor, input_tensor, group=None, async_op=False):
    try:
        import oneccl_bindings_for_pytorch  # noqa
    except ImportError as e:
        raise RuntimeError("oneccl_bindings_for_pytorch is not installed!") from e
    if not dist.is_initialized():
        dist.init_process_group("ccl")
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
