import torch
from typing import List
import os
import intel_extension_for_pytorch as ipex
import torch.distributed as dist
from torch.distributed import ReduceOp


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


USE_SHM_ALLREDUCE = -1


def all_reduce_cpu(t: torch.Tensor, op=ReduceOp.SUM, group=None, async_op=False):
    pg = (
        torch.distributed.distributed_c10d._get_default_group()
        if group is None
        else group
    )
    global USE_SHM_ALLREDUCE
    if USE_SHM_ALLREDUCE == -1:
        word_size = torch.distributed.get_world_size(pg)
        local_size = get_int_from_env(
            [
                "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "MV2_COMM_WORLD_LOCAL_SIZE",
                "LOCAL_WORLD_SIZE",
            ],
            -1,
        )
        if local_size >= 0 and local_size == word_size:
            USE_SHM_ALLREDUCE = 1
        else:
            USE_SHM_ALLREDUCE = -1

    if (
        USE_SHM_ALLREDUCE == 1
        and async_op is False
        and op is ReduceOp.SUM
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):

        ipex._C.tpp_shm_allreduce(t, pg)
        return t
    else:
        return dist.all_reduce(t, op, group, async_op)


def all_gather_cpu(
    t_list: List[torch.Tensor], t: torch.Tensor, group=None, async_op=False
):
    return dist.all_gather(t_list, t, group, async_op)


def all_gather_into_tensor_cpu(output_tensor, input_tensor, group=None, async_op=False):
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
