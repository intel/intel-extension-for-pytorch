import contextlib
import dataclasses
import functools
import hashlib
import sympy

from torch._dynamo.utils import dynamo_timed

from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import IndentedBuffer, PythonPrinter
from torch._inductor.codegen.wrapper import (
    MemoryPlanningState,
    MemoryPlanningLine,
    WrapperCodeGen,
)
from torch._inductor.utils import cache_on_self, get_benchmark_name
from .. import codecache

pexpr = PythonPrinter().doprint


@dataclasses.dataclass
class EnterXPUDeviceContextManagerLine:
    device_idx: int
    first_time: bool

    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if V.graph.cpp_wrapper:
            raise NotImplementedError
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with torch.xpu._DeviceGuard({self.device_idx}):")
            device_cm_stack.enter_context(code.indent())
            code.writeline(
                f"torch.xpu.set_device({self.device_idx}) # no-op to ensure context"
            )


class ExitXPUDeviceContextManagerLine:
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if not V.graph.cpp_wrapper:
            device_cm_stack.close()


class XPUTritonWrapperCodeGen(WrapperCodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import intel_extension_for_pytorch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile

                from torch import empty_strided, device
                from {codecache.__name__} import XPUAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = XPUAsyncCompile()

            """
        )

    @cache_on_self
    def write_triton_header_once(self):
        self.header.splice(
            """
            import triton
            import triton.language as tl
            from torch._inductor.triton_heuristics import grid, start_graph, end_graph
            from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
            """
        )

    def write_prefix(self):
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline("torch.xpu.synchronize()")
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()

    def write_get_raw_stream(self, index):
        self.write_triton_header_once()
        name = f"stream{index}"
        self.writeline(f"{name} = get_xpu_stream({index})")
        return name

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(
            EnterXPUDeviceContextManagerLine(device_idx, self.first_device_guard)
        )
        self.first_device_guard = False

    def codegen_device_guard_exit(self):
        self.lines.append(ExitXPUDeviceContextManagerLine())

    @dynamo_timed
    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)

        out_names = V.graph.get_output_names()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.write_triton_header_once()
                self.wrapper_call.writeline("start_graph()")

            while (
                self.lines
                and isinstance(self.lines[-1], MemoryPlanningLine)
                # TODO: this seems legit, NullLine has no node
                and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
            ):
                # these lines will be pointless
                self.lines.pop()

            # codegen allocations in two passes
            planning_state = MemoryPlanningState()
            for i in range(len(self.lines)):
                if isinstance(self.lines[i], MemoryPlanningLine):
                    self.lines[i] = self.lines[i].plan(planning_state)

            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(
                    line,
                    (
                        EnterXPUDeviceContextManagerLine,
                        ExitXPUDeviceContextManagerLine,
                    ),
                ):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline("torch.xpu.synchronize()")

            if config.profile_bandwidth:
                self.wrapper_call.writeline("end_graph()")

            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def benchmark_compiled_module(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(
                        name, shape, stride, value.get_device(), value.get_dtype()
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(
                f"return print_performance(lambda: {call_str}, times=times, repeat=repeat, device='xpu')"
            )

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        self.benchmark_compiled_module(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(
                [
                    "from intel_extension_for_pytorch._inductor.wrapper_benchmark import compiled_module_main",
                    f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)",
                ]
            )
