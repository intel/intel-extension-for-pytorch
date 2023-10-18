from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.triton_foreach import ForeachKernel


class XPUForeachKernel(ForeachKernel):
    def __init__(self):
        super().__init__()

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        code.splice(
            """
                import triton
                import triton.language as tl
                from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import foreach
                from torch._inductor.utils import instance_descriptor
                from torch._inductor import triton_helpers
            """
        )
        argdefs, _, _ = self.args.python_argdefs()
        code.writeline(self.jit_line())
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")

        with code.indent():
            code.splice("xpid = tl.program_id(0)")
            if self.blocking_2d:
                code.splice("ypid = tl.program_id(1)")
                code.splice(f"XBLOCK: tl.constexpr = {self.block_size_2d}")
                code.splice(f"YBLOCK: tl.constexpr = {self.block_size_2d}")
            else:
                code.splice(f"XBLOCK: tl.constexpr = {self.block_size_1d}")

            for sub_kernel in self.sub_kernels:
                assert len(sub_kernel.numels) <= 3
                # TODO mlazos: support dynamic shapes
                numel_ind = 0 if not self.blocking_2d else 1
                self.codegen_pid_range(code, int(sub_kernel.numels[numel_ind]))
                with code.indent():
                    if self.blocking_2d:
                        code.splice(f"ynumel = {sub_kernel.numels[0]}")
                        code.splice(f"xnumel = {sub_kernel.numels[1]}")
                    else:
                        code.splice(f"xnumel = {sub_kernel.numels[0]}")

                    sub_kernel.codegen_body()
                    code.splice(sub_kernel.body)

            code.splice("else:")
            with code.indent():
                code.splice("pass")

        return code.getvalue()
