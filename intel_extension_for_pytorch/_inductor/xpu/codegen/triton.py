import itertools
import sympy
import logging

import torch  # noqa
import torch._logging

from functools import lru_cache
from torch._dynamo.utils import counters
from torch._inductor.ir import ReductionHint
from torch._inductor.codecache import code_hash

from torch._inductor.virtualized import V
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._inductor.codegen.common import (
    IndentedBuffer,
    PythonPrinter,
    SizeArg,
)
from torch._inductor.codegen.triton import (
    signature_of,
    config_of,
    TritonKernel,
    TritonScheduling,
    EnableReduction,
    DisableReduction,
)
from torch._inductor import config, scheduler
from torch._inductor.codegen.triton_utils import signature_to_meta

from torch._inductor.utils import next_power_of_2, Placeholder
from torch.utils._triton import has_triton_package
from torch._inductor.scheduler import BaseSchedulerNode
from typing import cast


pexpr = PythonPrinter().doprint
log = logging.getLogger(__name__)


@lru_cache(None)
def gen_attr_descriptor_import():
    """
    import AttrsDescriptor if the triton version is new enough to have this
    class defined.
    """
    if not has_triton_package():
        return ""

    import triton.compiler.compiler

    if hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        return "from triton.compiler.compiler import AttrsDescriptor"
    else:
        return ""


@lru_cache(None)
def gen_common_triton_imports():
    imports = IndentedBuffer()
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)

    imports.splice(
        """
        from torch._inductor import triton_helpers, triton_heuristics
        from torch._inductor.ir import ReductionHint, TileHint
        from torch._inductor.triton_helpers import libdevice, math as tl_math
        from torch._inductor.triton_heuristics import AutotuneHint
        from torch._inductor.utils import instance_descriptor
        """
    )
    return imports.getvalue()


class XPUTritonKernel(TritonKernel):
    def __init__(
        self,
        *groups,
        index_dtype,
        mutations=None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        min_elem_per_thread=0,
        disable_persistent_reduction=False,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache=pid_cache,
            reduction_hint=reduction_hint,
            min_elem_per_thread=min_elem_per_thread,
            disable_persistent_reduction=disable_persistent_reduction,
        )

    def codegen_kernel_benchmark(self, num_gb, grid=None):
        result = IndentedBuffer()
        argdefs, call_args, signature = self.args.python_argdefs()

        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        if grid is None:
            grid = []
            extra_args = []
            extra_args_str = None
            for tree in self.active_range_trees():
                expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                extra_args.append(expr)
                if tree.prefix != "r":
                    grid.append(expr)
            if self.need_numel_args():
                extra_args_str = ", ".join(map(str, extra_args)) + ", "
            else:
                extra_args_str = ""
            grid_arg = f"{extra_args_str}grid=grid({', '.join(grid)})"
        else:
            grid_arg = f"grid={grid}"
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                stream_name = f"stream{index}"
                result.writeline(f"{stream_name} = get_raw_stream({index})")
                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, {grid_arg}, stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                result.writeline(
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {grid_arg})"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline(
                "from torch._inductor.utils import get_num_bytes, do_bench"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = do_bench(lambda: call(args), rep=40, fast_flush=True)"
            )
            result.writeline(f"num_gb = {num_gb}")
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)

        if not self.inside_reduction:
            size_hints.pop()

        heuristics = self._get_heuristic()

        if name is None:
            code.splice(gen_common_triton_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
        }

        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "no_x_dim": self.no_x_dim,
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),
        }
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(
                sizearg, size_dtype=self.index_dtype
            )
            argdefs.append(f"{tree.prefix}numel")
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]

        self.triton_meta = triton_meta

        for tree in self.range_trees:
            if tree.prefix == "r" and self.persistent_reduction:
                # RBLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()


class XPUTritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def codegen_node_schedule(
        self, node_schedule, buf_accesses, numel, reduction_numel
    ):
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        is_split_scan = any(
            isinstance(node, BaseSchedulerNode) and node.is_split_scan()
            for node in node_schedule
        )
        # TODO : DO we support TritonSplitScanKernel?
        kernel_type = TritonSplitScanKernel if is_split_scan else XPUTritonKernel
        kernel_args = tiled_groups
        kernel_kwargs = {
            "reduction_hint": reduction_hint_val,
            "mutations": mutations,
            "index_dtype": index_dtype,
        }

        kernel = kernel_type(
            *kernel_args,
            **kernel_kwargs,
        )

        kernel.buf_accesses = buf_accesses

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        if kernel.persistent_reduction and config.triton.multi_kernel:
            kernel2 = TritonKernel(
                *kernel_args,
                **kernel_kwargs,
                disable_persistent_reduction=True,
            )
            self.codegen_node_schedule_with_kernel(node_schedule, kernel2)
            with V.set_kernel_handler(kernel2):
                src_code2 = kernel2.codegen_kernel()
            kernel_name2 = self.define_kernel(src_code2, node_schedule)
            kernel2.kernel_name = kernel_name2
            kernel2.code_hash = code_hash(src_code2)

            final_kernel = MultiKernel([kernel, kernel2])
        else:
            final_kernel = kernel  # type: ignore[assignment]

        with V.set_kernel_handler(final_kernel):
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(final_kernel.kernel_name)
        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def codegen_foreach(self, foreach_node):
        from .triton_foreach import XPUForeachKernel

        for partitions_with_metadata in XPUForeachKernel.horizontal_partition(
            foreach_node.get_subkernel_nodes(), self
        ):
            kernel = XPUForeachKernel()
            for nodes, tiled_groups, numel, rnumel in partitions_with_metadata:
                node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
                (
                    reduction_hint_val,
                    mutations,
                    index_dtype,
                ) = self.get_kernel_args(node_schedule, numel, rnumel)

                subkernel = kernel.create_sub_kernel(
                    *tiled_groups,
                    reduction_hint=reduction_hint_val,
                    mutations=mutations,
                    index_dtype=index_dtype,
                )

                self.codegen_node_schedule_with_kernel(
                    node_schedule,
                    subkernel,
                )

                with V.set_kernel_handler(subkernel):
                    for node in node_schedule:
                        if node not in (EnableReduction, DisableReduction):
                            node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, [foreach_node])
            self.codegen_comment([foreach_node])
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.scheduler.free_buffers()
