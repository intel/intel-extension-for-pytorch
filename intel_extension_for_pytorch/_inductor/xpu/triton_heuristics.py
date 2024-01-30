import copy
import functools
import json
import logging
import operator
import os.path
import re
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch

from torch._inductor import config
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_heuristics import AutotuneHint  # noqa
from torch._inductor.utils import get_num_bytes, create_bandwidth_info_str
from torch._dynamo.utils import dynamo_timed, get_first_attr
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.utils import has_triton_package

from .utils import do_bench, has_triton


log = logging.getLogger(__name__)


if has_triton_package():
    import triton
    from triton import Config
    from triton.runtime.jit import KernelInterface

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        warnings.warn("XPU: Import error on ASTSource, if this is not the case, \
                      please comment out this")
        ASTSource = None
else:
    Config = object
    triton = None
    KernelInterface = object
    OutOfResources = object
    ASTSource = None



if has_triton():
    import triton
    from triton import Config
    from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
else:
    get_xpu_stream = None

from torch._inductor.triton_heuristics import (
    CachingAutotuner,
    _find_names,
    collected_calls,
    autotune_hints_to_configs,
    disable_pointwise_autotuning,
    HeuristicType,
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

    def __init__(
        self,
        fn,
        meta,
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
    ):
        super().__init__(
            fn,
            meta,
            configs,
            save_cache_hook,
            mutated_arg_names,
            heuristic_type,
            size_hints,
        )

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Dict):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = config.triton.assert_indirect_indexing
        compile_meta["device_type"] = "xpu"

        if warm_cache_only_with_cc:
            cc = warm_cache_only_with_cc
        else:
            # Use device_type 'cuda' for both cuda and hip devices to retrieve
            # the compute capability.
            device_type = "xpu"
            device_id = compile_meta["device"]
            device_interface = get_interface_for_device(device_type)
            device = torch.device(device_type, device_id)
            cc = device_interface.get_compute_capability(device)

        compile_meta["cc"] = cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    compile_meta["configs"][0],
                ),
            )

            target = (compile_meta["device_type"], cc)
            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta

        if warm_cache_only_with_cc:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # load binary to the correct device
        with torch.xpu.device(compile_meta["device"]):
            # need to initialize context
            torch.xpu.synchronize(torch.xpu.current_device())
            binary = triton.compile(*compile_args, **compile_kwargs)
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.xpu.set_device,
            "current_device": torch.xpu.current_device,
        }

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")
        scope["function"] = get_first_attr(binary, "function", "cu_function")
        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )
        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )
        scope["shared"] = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid


                runner(grid_0, grid_1, grid_2, num_warps,
                            *cta_args, shared,
                            stream, function, None, None, None,
                            {', '.join(call_args)})
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = getattr(binary, "shared", None)
        launcher.store_cubin = False
        # store this global varible to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return launcher

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        if launcher.n_spills > config.triton.spill_threshold:
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        stream = get_xpu_stream(torch.xpu.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )

            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)


class XPUDebugAutotuner(XPUCachingAutotuner):
    def __init__(self, *args, regex_filter="", **kwargs):
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, grid, stream, **kwargs):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=lambda x: len(x))}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream, **kwargs)
        (launcher,) = self.launchers

        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid, **kwargs)
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
            gb_per_s = num_gb / (ms / 1e3)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            ms, num_gb, gb_per_s, kernel_name = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(
            create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}")
        )


def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    meta,
    heuristic_type,
    filename=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]

    # on disk caching logic
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            with open(cache_filename, "w") as fd:
                fd.write(
                    json.dumps(
                        {
                            **cfg.kwargs,
                            "num_warps": cfg.num_warps,
                            "num_stages": cfg.num_stages,
                            "configs_hash": configs_hash,
                            "found_by_coordesc": found_by_coordesc,
                        }
                    )
                )
            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, cache_filename)

    else:
        save_cache_hook = None

    mutated_arg_names = meta.pop("mutated_arg_names", ())

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if config.profile_bandwidth:
            return XPUDebugAutotuner(
                fn,
                meta=meta,
                regex_filter=config.profile_bandwidth_regex,
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
            )
        return XPUCachingAutotuner(
            fn,
            meta=meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
        )

    return decorator


def pointwise(size_hints, meta, tile_hint=None, filename=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        meta.get("autotune_hints", set()), size_hints, bs
    )

    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, bs)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            return cached_autotune(
                size_hints,
                [
                    triton_config(size_hints, bs, num_elements_per_warp=256),
                    triton_config(size_hints, bs // 2, num_elements_per_warp=64),
                    *hinted_configs,
                ],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, 32, 32)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 64, 64),  # ~8% better for fp16
                triton_config(size_hints, 256, 16),
                triton_config(size_hints, 16, 256),
                triton_config(size_hints, bs, 1),
                triton_config(size_hints, 1, bs),
                *hinted_configs,
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, 16, 16, 16)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, bs, 1, 1),
                triton_config(size_hints, 1, bs, 1),
                triton_config(size_hints, 1, 1, bs),
                *hinted_configs,
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    """args to @triton.heuristics()"""
    assert meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048)
        )
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
        )
        if config.max_autotune or config.max_autotune_pointwise:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune(
                size_hints,
                [contiguous_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune(
                size_hints,
                [outer_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune(
                size_hints,
                [tiny_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config_reduction(size_hints, 32, 128)],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                contiguous_config,
                outer_config,
                tiny_config,
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
                # halve the XBLOCK/RBLOCK compared to outer_config
                # TODO: this may only be beneficial when each iteration of the reduciton
                # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
                triton_config_reduction(size_hints, 64, 4, num_warps=8),
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.REDUCTION,
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
    for c in configs:
        # we don't need RBLOCK for persistent reduction
        c.kwargs.pop("RBLOCK")

    if disable_pointwise_autotuning():
        configs = configs[:1]

    return cached_autotune(
        size_hints,
        configs,
        meta=meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def template(num_stages, num_warps, meta, filename=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        meta=meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def foreach(meta, num_warps, filename=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        meta=meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
