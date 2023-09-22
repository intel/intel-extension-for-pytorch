import itertools

import torch  # noqa

from torch._dynamo.utils import counters
from torch._inductor.ir import ReductionHint

from torch._inductor.virtualized import V

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
)
from torch._inductor import config, scheduler
from torch._inductor.codegen.triton_utils import signature_to_meta


pexpr = PythonPrinter().doprint


class XPUTritonKernel(TritonKernel):

    def __init__(
        self,
        *groups,
        index_dtype,
        mutations=None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache=pid_cache,
            reduction_hint=reduction_hint,
        )

    def codegen_kernel_benchmark(self):
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
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # noqa: B950 line too long
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
        grid = []
        extra_args = []
        extra_args_str = None
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f"with torch.xpu._DeviceGuard({index}):")
            with result.indent():
                result.writeline(
                    f"torch.xpu.set_device({index})"
                )  # no-op to ensure context
                for tree in self.range_trees:
                    expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                    if tree.prefix != "r" or self.inside_reduction:
                        extra_args.append(expr)
                    if tree.prefix != "r":
                        grid.append(expr)

                stream_name = f"stream{index}"
                result.writeline(f"{stream_name} = get_xpu_stream({index})")
                extra_args_str = ", ".join(map(str, extra_args)) + ", "
                result.writeline(
                    f"KERNEL_NAME.run(*args, {extra_args_str}grid=grid({', '.join(grid)}), stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            result.writeline(f"with torch.xpu._DeviceGuard({index}):")
            with result.indent():
                result.writeline(
                    f"torch.xpu.set_device({index})"
                )  # no-op to ensure context
                result.writeline(
                    f"return KERNEL_NAME.benchmark_all_configs(*args, {extra_args_str}grid=grid({', '.join(grid)}))"
                )

        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline("from torch._inductor.utils import get_num_bytes")
            result.writeline("from intel_extension_for_pytorch._inductor.utils import do_bench")
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = do_bench(lambda: call(args), rep=40, fast_flush=True)"
            )
            result.writeline(
                f"num_gb = get_num_bytes(*args, num_in_out_args={ninplace_args}) / 1e9"
            )
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2

        code = IndentedBuffer()

        size_hints = [
            next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels
        ]
        if self.persistent_reduction:
            assert self.inside_reduction
            heuristics = "persistent_reduction"
        elif self.inside_reduction:
            heuristics = "reduction"
        else:
            size_hints.pop()
            heuristics = "pointwise"

        if name is None:
            code.splice(
                f"""
                    import triton
                    import triton.language as tl
                    from torch._inductor.ir import ReductionHint
                    from torch._inductor.ir import TileHint
                    from intel_extension_for_pytorch._inductor.triton_heuristics import AutotuneHint, {heuristics}
                    from torch._inductor.utils import instance_descriptor
                    from torch._inductor import triton_helpers
                """
            )
            if config.benchmark_kernel:
                code.splice(
                    """
                        from torch._dynamo.testing import rand_strided
                        import torch
                        from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
                        from torch._inductor.triton_heuristics import grid
                    """
                )

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if its in sizevars replacements
        for i, arg in enumerate(signature):
            if (
                isinstance(arg, SizeArg)
                and arg.expr in V.graph.sizevars.inv_precomputed_replacements
            ):
                signature[i] = SizeArg(
                    arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr]
                )

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=self.index_dtype),
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
            "mutated_arg_names": mutated_args,
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": "DESCRIPTIVE_KRNL_NAME",
        }

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta["signature"][len(argdefs)] = signature_of(
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

        for tree in self.range_trees:
            if tree.prefix == "r" and (
                not self.inside_reduction or self.persistent_reduction
            ):
                continue
            if tree.prefix == "x" and self.no_x_dim:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    meta={triton_meta!r}
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
                @{heuristics}(size_hints={size_hints!r}, {tile_hint}filename=__file__, meta={triton_meta!r})
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")
        self.codegen_body()
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark())

        if name is not None:
            return code.getvalue()

        return code.getvalue()


class XPUTritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        kernel = XPUTritonKernel(
            *tiled_groups,
            reduction_hint=reduction_hint_val,
            mutations=mutations,
            index_dtype=index_dtype,
        )

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name)

        if config.warn_mix_layout:
            kernel.warn_mix_layout(kernel_name)

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

    def codegen_sync(self):
        V.graph.wrapper_code.writeline("torch.xpu.synchronize()")

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
                self.codegen_node_schedule_with_kernel(
                    node_schedule,
                    kernel.create_sub_kernel(
                        *tiled_groups,
                        reduction_hint=reduction_hint_val,
                        mutations=mutations,
                        index_dtype=index_dtype,
                    ),
                )

            src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, [foreach_node])
            self.codegen_comment([foreach_node])
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.scheduler.free_buffers()
