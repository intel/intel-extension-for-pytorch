import copy
import functools
import json
import operator
import os.path
from typing import List

import torch

from torch._inductor import config
from torch._inductor.ir import ReductionHint, TileHint

from ..utils import do_bench, has_triton
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream

if has_triton():
    import triton
    from triton import cdiv, Config, next_power_of_2
    from triton.runtime.jit import KernelInterface
    # Here we override these objects for leveraging more codes related with them from PyTorch.
    import torch._inductor.triton_ops.autotune as _autotune
    _autotune.cdiv = cdiv
    _autotune.Config = Config
    _autotune.next_power_of_2 = next_power_of_2
    _autotune.KernelInterface = KernelInterface

# NOTE: we have to import these objects after overriding in PyTorch to guarantee their correctness.
from torch._inductor.triton_ops.autotune import (
    CachingAutotuner,
    hash_configs,
    load_cached_autotuning,
    unique_configs,
    triton_config,
    triton_config_reduction,
)

class XPUCachingAutotuner(CachingAutotuner):

    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, meta, configs, save_cache_hook, mutated_arg_names):
        super().__init__(fn, meta, configs, save_cache_hook, mutated_arg_names)

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["device_type"] = "xpu"
        if warm_cache_only_with_cc:
            triton.compile(
                self.fn,
                warm_cache_only=True,
                cc=warm_cache_only_with_cc,
                **compile_meta,
            )
            return

        # load binary to the correct device
        with torch.xpu.device(compile_meta["device"]):
            # need to initialize context
            torch.xpu.synchronize(torch.xpu.current_device())
            binary = triton.compile(
                self.fn,
                **compile_meta,
            )

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = list(self.fn.arg_names)
        while def_args and def_args[-1] in cfg.kwargs:
            def_args.pop()

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.xpu.set_device,
            "current_device": torch.xpu.current_device,
        }
        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid
                bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,
                            stream, bin.cu_function, None, None, bin,
                            {', '.join(call_args)})
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        return launcher

    def bench(self, launcher, *args, grid):
        """Measure the performance of a given launcher"""
        stream = get_xpu_stream(torch.xpu.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**zip(self.arg_names, args), **launcher.config.kwargs}
                )
            launcher(
                *args,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)


def cached_autotune(
    configs: List[Config],
    meta,
    filename=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename

    # on disk caching logic
    if filename is not None and len(configs) > 1:
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg):
            with open(cache_filename, "w") as fd:
                fd.write(json.dumps({**cfg.kwargs, "configs_hash": configs_hash}))

    else:
        save_cache_hook = None

    mutated_arg_names = meta.pop("mutated_arg_names", ())

    def decorator(fn):
        return XPUCachingAutotuner(
            fn,
            meta=meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
        )

    return decorator


def pointwise(size_hints, meta, tile_hint=None, filename=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    if len(size_hints) == 1:
        return cached_autotune([triton_config(size_hints, bs)], meta=meta)
    if len(size_hints) == 2:
        if (
            not config.triton.autotune_pointwise or tile_hint == TileHint.SQUARE
        ) and not config.max_autotune:
            return cached_autotune([triton_config(size_hints, 32, 32)], meta=meta)
        return cached_autotune(
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 64, 64),  # ~8% better for fp16
                triton_config(size_hints, 256, 16),
                triton_config(size_hints, 16, 256),
                triton_config(size_hints, bs, 1),
                triton_config(size_hints, 1, bs),
            ],
            meta=meta,
            filename=filename,
        )
    if len(size_hints) == 3:
        if not config.triton.autotune_pointwise:
            return cached_autotune([triton_config(size_hints, 16, 16, 16)], meta=meta)
        return cached_autotune(
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, bs, 1, 1),
                triton_config(size_hints, 1, bs, 1),
                triton_config(size_hints, 1, 1, bs),
            ],
            meta=meta,
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    """args to @triton.heuristics()"""
    assert meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048), num_stages=1
        )
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
        )
        if config.max_autotune:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune([contiguous_config], meta=meta)
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune([outer_config], meta=meta)
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune([tiny_config], meta=meta)
        if not config.triton.autotune_pointwise:
            return cached_autotune(
                [triton_config_reduction(size_hints, 32, 128)], meta=meta
            )
        return cached_autotune(
            [
                contiguous_config,
                outer_config,
                tiny_config,
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
            ],
            meta=meta,
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def persistent_reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    xnumel, rnumel = size_hints

    configs = [
        triton_config_reduction(size_hints, xblock, rnumel)
        for xblock in (1, 8, 32, 128)
        if rnumel * xblock <= 4096 and xblock <= xnumel
    ]

    # TODO(jansel): we should be able to improve these heuristics
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [
            triton_config_reduction(
                size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
            )
        ]

    return cached_autotune(
        configs,
        meta=meta,
        filename=filename,
    )


def template(num_stages, num_warps, meta, filename=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)], meta=meta
    )
