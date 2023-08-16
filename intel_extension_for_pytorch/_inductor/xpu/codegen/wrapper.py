import contextlib
import dataclasses
import functools
import hashlib

from torch._dynamo.utils import dynamo_timed

from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import IndentedBuffer, PythonPrinter
from torch._inductor.codegen.wrapper import (
    MemoryPlanningState,
    MemoryPlanningLine,
    WrapperCodeGen,
)
from .. import codecache
from ..utils import has_triton

pexpr = PythonPrinter().doprint


@dataclasses.dataclass
class EnterXPUDeviceContextManagerLine:
    device_idx: int

    def codegen(self, code: IndentedBuffer):
        # Note _DeviceGuard has less overhead than device, but only accepts
        # integers
        code.writeline(f"with torch.xpu._DeviceGuard({self.device_idx}):")


class ExitXPUDeviceContextManagerLine:
    pass


class XPUTritonWrapperCodeGen(WrapperCodeGen):
    """
    The outer wrapper that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self.header = IndentedBuffer()
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                from torch import empty_strided, as_strided, device
                from {codecache.__name__} import XPUAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                async_compile = XPUAsyncCompile()

            """
        )

        if has_triton():
            self.header.splice(
                """
                import triton
                import triton.language as tl
                from torch._inductor.triton_ops.autotune import grid
                from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
                """
            )

        for name, value in V.graph.constants.items():
            # include a hash so our code cache gives different constants different files
            hashed = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
            self.header.writeline(f"{name} = None  # {hashed}")

        self.write_get_xpu_stream = functools.lru_cache(None)(
            self.write_get_xpu_stream
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
            for name in V.graph.randomness_seeds:
                self.prefix.writeline(
                    f"torch.randint(2**31, size=(), dtype=torch.int64, out={name})"
                )
            V.graph.sizevars.codegen(self.prefix, V.graph.graph_inputs)

    def write_get_xpu_stream(self, index):
        name = f"stream{index}"
        self.writeline(f"{name} = get_xpu_stream({index})")
        return name

    def codegen_device_guard_enter(self, device_idx):
        self.lines.append(EnterXPUDeviceContextManagerLine(device_idx))

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
                self.wrapper_call.writeline(
                    "from torch.profiler import record_function"
                )
                self.wrapper_call.writeline(
                    "with record_function('inductor_wrapper_call'):"
                )
                stack.enter_context(self.wrapper_call.indent())
            while (
                self.lines
                and isinstance(self.lines[-1], MemoryPlanningLine)
                and self.lines[-1].node.name not in out_names
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
                elif isinstance(line, EnterXPUDeviceContextManagerLine):
                    line.codegen(self.wrapper_call)
                    device_cm_stack.enter_context(self.wrapper_call.indent())
                    self.wrapper_call.writeline(
                        f"torch.xpu.set_device({line.device_idx}) # no-op to ensure context"
                    )
                elif isinstance(line, ExitXPUDeviceContextManagerLine):
                    device_cm_stack.close()
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline("torch.xpu.synchronize()")
            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvalue()

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{V.graph.sizevars.codegen_benchmark_shape_tuple(shape)}, "
                f"{V.graph.sizevars.codegen_benchmark_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from intel_extension_for_pytorch._inductor.xpu.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            for name, value in V.graph.graph_inputs.items():
                shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                add_fake_input(
                    name, shape, stride, value.get_device(), value.get_dtype()
                )

            output.writeline(
                f"print_performance(lambda: call([{', '.join(V.graph.graph_inputs.keys())}]))"
            )
