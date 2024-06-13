import sys
from intel_extension_for_pytorch.transformers.models.cpu.distributed.dist import (  # noqa
    all_reduce_cpu,
    all_gather_cpu,
    all_gather_into_tensor_cpu,
)
import torch
from torch.distributed import Backend, default_pg_timeout, Store, ReduceOp
import torch.distributed as dist
from datetime import timedelta
from typing import Union, Optional, Any


def _get_function_from_device(device_type: str, f):
    assert device_type in [
        "cpu",
        "xpu",
    ], "The device is not in the supported device list."
    target_f_name = f.__name__ + "_" + device_type
    assert hasattr(
        sys.modules[__name__], target_f_name
    ), f"Target function {f.__name__} on {device_type} haven't implemented yet."
    target_f = getattr(sys.modules[__name__], target_f_name)
    return target_f


def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: timedelta = default_pg_timeout,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[torch.device] = None,
):
    """
    Initialize the default distributed process group.
    This will also initialize the distributed package.
    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.
    If neither is specified, ``init_method`` is assumed to be "env://".
    Args:
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            ``nccl``, ``ccl``, and ``ucc``. If the backend is not provided, then both a ``gloo``
            and ``nccl`` backend will be created, see notes below for how multiple
            backends are managed. This field can be given as a lowercase string
            (e.g., ``"gloo"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
            multiple processes per machine with ``nccl`` backend, each process
            must have exclusive access to every GPU it uses, as sharing GPUs
            between processes can result in deadlocks. ``ucc`` backend is
            experimental.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.
        group_name (str, optional, deprecated): Group name. This argument is ignored
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. As of now, the only
            options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            the nccl backend can pick up high priority cuda streams when
            there're compute kernels waiting.
        device_id (torch.device, optional): a single, specific device
            to "bind" this process to, allowing for backend-specific
            optimizations.  Currently this has two effects, only under
            NCCL: the communicator is immediately formed (calling
            ``ncclCommInit*`` immediately rather than the normal lazy
            call) and sub-groups will use ``ncclCommSplit`` when
            possible to avoid unnecessary overhead of group creation. If you
            want to know NCCL initialization error early, you can also use this
            field.
    .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.
    .. note:: Support for multiple backends is experimental. Currently when no backend is
        specified, both ``gloo`` and ``nccl`` backends will be created. The ``gloo`` backend
        will be used for collectives with CPU tensors and the ``nccl`` backend will be used
        for collectives with CUDA tensors. A custom backend can be specified by passing in
        a string with format "<device_type>:<backend_name>,<device_type>:<backend_name>", e.g.
        "cpu:gloo,cuda:custom_backend".
    .. note:: To enable backend ``ccl``, oneccl_bindings_for_pytorch needs to be installed
        and it will be imported automatically.
    """
    if backend == "ccl":
        try:
            import oneccl_bindings_for_pytorch  # noqa
        except ImportError as e:
            raise RuntimeError("oneccl_bindings_for_pytorch is not installed!")
    return dist.init_process_group(
        backend,
        init_method,
        timeout,
        world_size,
        rank,
        store,
        group_name,
        pg_options,
        device_id,
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.
    After the call ``tensor`` is going to be bitwise identical in all processes.
    Complex tensors are supported.
    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1
        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j]) # Rank 0
        tensor([4.+4.j, 6.+6.j]) # Rank 1
    """
    f = _get_function_from_device(tensor.device.type, all_reduce)
    return f(tensor, op, group, async_op)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    """
    Gathers tensors from the whole group in a list.
    Complex tensors are supported.
    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]
        >>> tensor_list
        [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2]), tensor([3, 4])] # Rank 0
        [tensor([1, 2]), tensor([3, 4])] # Rank 1
        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [torch.zeros(2, dtype=torch.cfloat) for _ in range(2)]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j]), tensor([0.+0.j, 0.+0.j])] # Rank 0 and 1
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 0
        [tensor([1.+1.j, 2.+2.j]), tensor([3.+3.j, 4.+4.j])] # Rank 1
    """
    f = _get_function_from_device(tensor.device.type, all_gather)
    return f(tensor_list, tensor, group, async_op)


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    """
    Gather tensors from all ranks and put them in a single output tensor.
    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            Different from the ``all_gather`` API, the input tensors in this
            API must have the same size across all ranks.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on XPU devices.
        >>> # We have two ranks.
        >>> device = torch.device(f'xpu:{rank}')
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device='xpu:0') # Rank 0
        tensor([3, 4], device='xpu:1') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device='xpu:0') # Rank 0
        tensor([1, 2, 3, 4], device='xpu:1') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device='xpu:0') # Rank 0
        tensor([[1, 2],
                [3, 4]], device='xpu:1') # Rank 1
    .. warning::
        The Gloo backend does not support this API.
    """
    f = _get_function_from_device(output_tensor.device.type, all_gather_into_tensor)
    return f(output_tensor, input_tensor, group, async_op)
