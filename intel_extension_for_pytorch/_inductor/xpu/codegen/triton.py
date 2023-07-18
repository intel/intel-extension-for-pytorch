import contextlib

import sympy

import torch  # noqa

from torch._dynamo import config as dynamo_config
from torch._inductor.ir import ReductionHint
from torch._inductor.optimize_indexing import indexing_dtype_strength_reduction

from torch._inductor.virtualized import V

from torch._inductor.codegen.common import (
    IndentedBuffer,
    PythonPrinter,
    SizeArg,
)
from torch._inductor.codegen.triton import (
    signature_of,
    config_of,
    TritonPrinter,
    TritonOverrides,
    TritonKernel,
    TritonScheduling,
    DisableReduction,
    EnableReduction,
)


class XPUTritonPrinter(TritonPrinter):
    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"tl.math.floor({self.paren(self._print(expr.args[0]))})"


texpr = XPUTritonPrinter().doprint
pexpr = PythonPrinter().doprint


class XPUTritonOverrides(TritonOverrides):
    """Map element-wise ops to XPU Triton backend"""

    @staticmethod
    def libdevice_abs(x):
        return f"tl.math.abs({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"tl.math.exp({x})"

    @staticmethod
    def exp2(x):
        return f"tl.math.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"tl.math.expm1({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"tl.math.sqrt({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"tl.math.cos({x})"

    @staticmethod
    def libdevice_sin(x):
        return f"tl.math.sin({x})"

    @staticmethod
    def lgamma(x):
        return f"tl.math.lgamma({x})"

    @staticmethod
    def erf(x):
        return f"tl.math.erf({x})"

    @staticmethod
    def cosh(x):
        return f"tl.math.cosh({x})"

    @staticmethod
    def sinh(x):
        return f"tl.math.sinh({x})"

    @staticmethod
    def acos(x):
        return f"tl.math.acos({x})"

    @staticmethod
    def acosh(x):
        return f"tl.math.acosh({x})"

    @staticmethod
    def asin(x):
        return f"tl.math.asin({x})"

    @staticmethod
    def asinh(x):
        return f"tl.math.asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"tl.math.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"tl.math.atan({x})"

    @staticmethod
    def atanh(x):
        return f"tl.math.atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"tl.math.copysign({x}, {y})"

    @staticmethod
    def erfc(x):
        return f"tl.math.erfc({x})"

    @staticmethod
    def hypot(x, y):
        return f"tl.math.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"tl.math.log10({x})"

    @staticmethod
    def nextafter(x, y):
        return f"tl.math.nextafter({x}, {y})"

    @staticmethod
    def rsqrt(x):
        return f"tl.math.rsqrt({x})"

    @staticmethod
    def log1p(x):
        return f"tl.math.log1p({x})"

    @staticmethod
    def tan(x):
        return f"tl.math.tan({x})"

    @staticmethod
    def tanh(x):
        return f"tl.math.tanh({x})"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1/(1 + tl.math.exp(-({x})))"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"tl.math.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"tl.math.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"tl.math.pow({a}, {b})"

    @staticmethod
    def libdevice_log(x):
        return f"tl.math.log({x})"

    @staticmethod
    def isinf(x):
        return f"tl.math.isinf({x})"

    @staticmethod
    def isnan(x):
        return f"tl.math.isnan({x})"

    @staticmethod
    def round(x):
        return f"tl.math.nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"tl.math.floor({x})"

    @staticmethod
    def trunc(x):
        return f"tl.math.trunc({x})"

    @staticmethod
    def ceil(x):
        return f"tl.math.ceil({x})"


class XPUTritonKernel(TritonKernel):
    overrides = XPUTritonOverrides
    sexpr = pexpr

    def __init__(
        self,
        *groups,
        mutations=None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
    ):
        super().__init__(*groups, mutations=mutations, pid_cache=pid_cache, reduction_hint=reduction_hint)

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
                    from intel_extension_for_pytorch._inductor.xpu.triton_ops.autotune import {heuristics}
                    from torch._inductor.utils import instance_descriptor
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
            if mutation in self.args.inplace_buffers:
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta = {
            "signature": dict(enumerate(map(signature_of, signature))),
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
            "mutated_arg_names": mutated_args,
        }

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta["signature"][len(argdefs)] = signature_of(sizearg)
                argdefs.append(f"{tree.prefix}numel")
                # constexpr version causes issues, see
                # https://github.com/pytorch/torchdynamo/pull/1362
                # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
                #     tree.numel
                # )
                # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
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
            if not dynamo_config.dynamic_shapes:
                self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("async_compile.triton('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''')")
        return wrapper.getvalue()

    def call_kernel(self, code, name: str):
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        grid = []
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = pexpr(tree.numel)
            else:
                expr = f"{name}_{tree.prefix}numel"
                code.writeline(f"{expr} = {pexpr(tree.numel)}")
            if tree.prefix != "r" or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)
        call_args = ", ".join(call_args)
        stream_name = code.write_get_xpu_stream(V.graph.scheduler.current_device.index)
        code.writeline(
            f"{name}.run({call_args}, grid=grid({', '.join(grid)}), stream={stream_name})"
        )


class XPUTritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reductions = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and n.is_reduction(),
                node_schedule,
            )
        )
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT

        mutations = set()
        for node in node_schedule:
            if hasattr(node, "get_mutations"):
                mutations.update(node.get_mutations())

        with XPUTritonKernel(
            *tiled_groups, reduction_hint=reduction_hint_val, mutations=mutations
        ) as kernel:
            stack = contextlib.ExitStack()
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    # TODO - mostly works but needs a couple fixes
                    if not dynamo_config.dynamic_shapes:
                        # TODO - use split ranges ?
                        indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

        src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, node_schedule)
        kernel.call_kernel(V.graph.wrapper_code, kernel_name)
        self.scheduler.free_buffers()

    def codegen_sync(self):
        V.graph.wrapper_code.writeline("torch.xpu.synchronize()")
