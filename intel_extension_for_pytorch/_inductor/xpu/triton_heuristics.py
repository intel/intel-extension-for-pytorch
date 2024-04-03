import copy
import functools
import hashlib
import json
import logging
import operator
import os.path
import re
from typing import Any, Callable, Dict, List, Optional

import torch

from torch._dynamo.utils import dynamo_timed, get_first_attr
from torch._inductor import config
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_heuristics import AutotuneHint  # noqa
from torch._inductor.utils import get_num_bytes, create_bandwidth_info_str, do_bench

from .utils import has_triton
from torch._C import _xpu_getCurrentRawStream as get_xpu_stream

from torch.utils._triton import has_triton_package


log = logging.getLogger(__name__)

if has_triton_package():
    import triton
    from triton import Config
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import KernelInterface

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None
else:
    Config = object
    triton = None
    KernelInterface = object
    OutOfResources = object
    ASTSource = None

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
        triton_meta,  # passed directly to triton
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # metadata not relevant to triton
        custom_kernel=False,  # whether the kernel is inductor-generated or custom
    ):
        super().__init__(
            fn,
            triton_meta,
            configs,
            save_cache_hook,
            mutated_arg_names,
            heuristic_type,
            size_hints,
            inductor_meta,
            custom_kernel,
        )

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Dict):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = config.assert_indirect_indexing

        compile_meta["device_type"] = "xpu"

        if warm_cache_only_with_cc:
            cc = warm_cache_only_with_cc
        else:
            # Use device_type 'cuda' for both cuda and hip devices to retrieve
            # the compute capability.
            device_type = self.device_type if torch.version.hip is None else "cuda"
            device_id = compile_meta["device"]
            device = torch.device(device_type, device_id)
            cc = self.gpu_device.get_compute_capability(device)

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
        with self.gpu_device.device(compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            self.gpu_device.synchronize(self.gpu_device.current_device())

            try:
                binary = triton.compile(*compile_args, **compile_kwargs)
            except Exception:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    compile_meta,
                )
                raise
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
            "launch_enter_hook": binary.launch_enter_hook,
            "launch_exit_hook": binary.launch_exit_hook,
            "metadata": binary.metadata,
            "torch": torch,
            "set_device": self.gpu_device.set_device,
            "current_device": self.gpu_device.current_device,
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
        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )
        scope["shared"] = binary_shared

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                runner(grid_0, grid_1, grid_2, num_warps,
                            *cta_args, shared,
                            stream, function,
                            launch_enter_hook,
                            launch_exit_hook,
                            metadata,
                            {', '.join(call_args)})
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = config.triton.store_cubin
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        # we don't skip configs wiht spilled registers when auto-tuning custom
        # (user-written) Triton kernels, as (i) we don't have any knowledge or
        # control over the kernel code; (ii) there is empirical evidence that
        # for some (complicated) custom Triton kernels, a register-spilling
        # config may yield the best latency.
        if not self.custom_kernel and launcher.n_spills > config.triton.spill_threshold:
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        stream = self.gpu_device.get_raw_stream(  # type: ignore[call-arg]
            self.gpu_device.current_device()
        )

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

    def run(self, *args, grid, stream):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=len)}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        (launcher,) = self.launchers

        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid)
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            num_gb = self.inductor_meta.get("kernel_num_gb", None)
            if num_gb is None:
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
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]
    inductor_meta = {} if inductor_meta is None else inductor_meta

    # on disk caching logic and/or remote caching
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        configs_hash = hash_configs(configs)

        cache_filename = None
        remote_cache = None
        remote_cache_key = None
        if config.use_autotune_local_cache:
            cache_filename = os.path.splitext(filename)[0] + ".best_config"
        if config.use_autotune_remote_cache or (
            config.is_fbcode()
            and torch._utils_internal.justknobs_check(
                "pytorch/autotune_remote_cache:enable"
            )
        ):
            backend_hash = inductor_meta.get("backend_hash", None)
            if backend_hash is not None:
                key = backend_hash + configs_hash + "autotune-best-config"
                key = hashlib.sha256(key.encode("utf-8")).hexdigest()

                try:
                    if config.is_fbcode():
                        remote_cache = (
                            triton.runtime.fb_memcache.FbMemcacheRemoteCacheBackend(
                                key, is_autotune=True
                            )
                        )
                    else:
                        remote_cache = triton.runtime.cache.RedisRemoteCacheBackend(key)
                except Exception:
                    remote_cache = None
                    log.warning("Unable to create a remote cache", exc_info=True)
                # we already sha256 hash the source contents
                remote_cache_key = os.path.basename(filename)
            else:
                log.debug(
                    "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
                )

        best_config = None
        if cache_filename is not None and os.path.exists(cache_filename):
            with open(cache_filename) as fd:
                best_config = json.loads(fd.read())
        elif remote_cache is not None and remote_cache_key is not None:
            cache_outs = remote_cache.get([remote_cache_key])
            cache_out = cache_outs.get(remote_cache_key, None)
            best_config = json.loads(cache_out) if cache_out else None

        best_config = load_cached_autotuning(best_config, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            data = json.dumps(
                {
                    **cfg.kwargs,
                    "num_warps": cfg.num_warps,
                    "num_stages": cfg.num_stages,
                    "configs_hash": configs_hash,
                    "found_by_coordesc": found_by_coordesc,
                }
            )
            if cache_filename is not None:
                with open(cache_filename, "w") as fd:
                    fd.write(data)
            if remote_cache is not None and remote_cache_key is not None:
                remote_cache.put(remote_cache_key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, cache_filename)

    else:
        save_cache_hook = None

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())

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
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=config.profile_bandwidth_regex,
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
            )
        return XPUCachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
        )

    return decorator


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert not inductor_meta.get("no_x_dim")

    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", set()), size_hints, bs
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, bs)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            return cached_autotune(
                size_hints,
                [
                    triton_config_with_settings(
                        size_hints, bs, num_elements_per_warp=256
                    ),
                    triton_config_with_settings(
                        size_hints, bs // 2, num_elements_per_warp=64
                    ),
                    *hinted_configs,
                ],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 32, 32)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 16, 16, 16)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def _reduction_configs(
    *, size_hints: List[int], inductor_meta: Dict[str, Any]
) -> List[Config]:
    reduction_hint = inductor_meta.get("reduction_hint", None)
    assert len(size_hints) == 2
    rnumel = size_hints[-1]

    contiguous_config = triton_config_reduction(
        size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048)
    )
    outer_config = triton_config_reduction(size_hints, 64, 8)
    tiny_config = triton_config_reduction(
        size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
    )
    if config.max_autotune or config.max_autotune_pointwise:
        pass  # skip all these cases
    elif reduction_hint == ReductionHint.INNER:
        return [contiguous_config]
    elif reduction_hint == ReductionHint.OUTER:
        return [outer_config]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return [tiny_config]
    if disable_pointwise_autotuning():
        return [triton_config_reduction(size_hints, 32, 128)]
    return [
        contiguous_config,
        outer_config,
        tiny_config,
        triton_config_reduction(size_hints, 64, 64),
        triton_config_reduction(size_hints, 8, 512),
        # halve the XBLOCK/RBLOCK compared to outer_config
        # TODO: this may only be beneficial when each iteration of the reduction
        # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
        triton_config_reduction(size_hints, 64, 4, num_warps=8),
    ]


def reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints = [1, *size_hints[1:]]

    assert triton_meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _reduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints = [1, *size_hints[1:]]

    xnumel, rnumel = size_hints

    configs = [
        triton_config_reduction(size_hints, xblock, rnumel)
        for xblock in (1, 8, 32, 128)
        if xblock == 1 or (rnumel * xblock <= 4096 and xblock <= xnumel)
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
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
