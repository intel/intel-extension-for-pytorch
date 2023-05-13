import collections
from typing import Any, Dict, Union

import intel_extension_for_pytorch
from torch.types import Device
from torch._utils import _get_device_index


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    sysman toolkit.

    .. note::
        :func:`~torch.xpu.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`xpu-memory-management` for
        more details about GPU memory management.
    """
    intel_extension_for_pytorch._C._emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns a dictionary of XPU memory allocator statistics for a
    given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``xpuMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``xpuMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns the result of :func:`~torch.xpu.memory_stats` as a nested dictionary."""
    device = _get_device_index(device, optional=True)
    return intel_extension_for_pytorch._C._memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "accumulated" (historical) stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return intel_extension_for_pytorch._C._resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "peak" stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return intel_extension_for_pytorch._C._resetPeakMemoryStats(device)


def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in sysman toolkit since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`xpu-memory-management` for more
        details about GPU memory management.
    """
    return memory_stats(device=device)["allocated_bytes.all.current"]


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GPU memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.xpu.reset_peak_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device)["allocated_bytes.all.peak"]


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the current GPU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device)["reserved_bytes.all.current"]


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GPU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.xpu.reset_peak_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device)["reserved_bytes.all.peak"]


def memory_snapshot():
    r"""Returns a snapshot of the XPU memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    return intel_extension_for_pytorch._C._memorySnapshot()


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return "{:7d} {}".format(sz, prefix)

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return "{:7d} {} ".format(cnt, prefix)

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch XPU memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} XPU OOMs: {num_ooms:<12d} | {_:6} xpuMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                " {:<21} | {} | {} | {} | {} ".format(
                    submetric_name,
                    formatter(current, current_prefval),
                    formatter(peak, peak_prefval),
                    formatter(allocated, allocated_prefval),
                    formatter(freed, freed_prefval),
                ),
            )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"
