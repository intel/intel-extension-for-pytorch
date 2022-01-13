import torch

class AutocastCPUTestLists(object):
    # Supplies ops and arguments for test_autocast_* in test/test_cpu.py
    def __init__(self, dev):
        super().__init__()
        n = 8
        # Utility arguments, created as one-element tuples
        pointwise0_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        pointwise1_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        pointwise2_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
        mat0_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat1_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
        mat2_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)

        dummy_dimsets = ((n,), (n, n), (n, n, n), (n, n, n, n), (n, n, n, n, n))

        dummy_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),)
                      for dimset in dummy_dimsets]

        dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
        conv_args_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),
                           torch.randn(dimset, dtype=torch.bfloat16, device=dev))
                          for dimset in dimsets]
        conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                           torch.randn(dimset, dtype=torch.float32, device=dev))
                          for dimset in dimsets]

        bias_fp32 = (torch.randn((n,), dtype=torch.float32, device=dev),)
        element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)
        pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        pointwise1_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
        mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
        mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

        dummy_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),)
                      for dimset in dummy_dimsets]
        # The lists below organize ops that autocast needs to test.
        # self.list_name corresponds to test_autocast_list_name in test/test_cpu.py.
        # Each op is associated with a tuple of valid arguments.

        # Some ops implement built-in type promotion.  These don't need autocasting,
        # but autocasting relies on their promotion, so we include tests to double-check.
        self.torch_expect_builtin_promote = [
            ("eq", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("ge", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("gt", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("le", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("lt", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("ne", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("add", pointwise0_fp32 + pointwise1_bf16, torch.float32),
            ("div", pointwise0_fp32 + pointwise1_bf16, torch.float32),
            ("mul", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ]
        self.methods_expect_builtin_promote = [
            ("__eq__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__ge__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__gt__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__le__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__lt__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__ne__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
            ("__add__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
            ("__div__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
            ("__mul__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ]
        # The remaining lists organize ops that autocast treats explicitly.
        self.torch_bf16 = [
            ("conv1d", conv_args_fp32[0]),
            ("conv2d", conv_args_fp32[1]),
            ("conv3d", conv_args_fp32[2]),
            ("conv_transpose1d", conv_args_fp32[0]),
            ("conv_transpose2d", conv_args_fp32[1]),
            ("conv_transpose3d", conv_args_fp32[2]),
            ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("mm", mat0_fp32 + mat1_fp32),
            ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32),
                         torch.randn((n, n, n), device=dev, dtype=torch.float32))),
            ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
            ("addbmm", mat0_fp32 + (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                                    torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ]
        self.torch_fp32 = [
        ]
        self.nn_bf16 = [
            ("linear", mat0_fp32 + mat1_fp32),
        ]
        self.fft_fp32 = [
        ]
        self.special_fp32 = [
        ]
        self.linalg_fp32 = [
            ("linalg_matrix_rank", dummy_bf16[2]),
        ]
        self.nn_fp32 = [
        ]
        self.torch_need_autocast_promote = [
            ("cat", (pointwise0_bf16 + pointwise1_fp32,)),
            ("stack", (pointwise0_bf16 + pointwise1_fp32,)),
            ("index_copy", (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16), 0, torch.tensor([0, 1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))),
        ]
        self.blacklist_non_float_output_pass_test = [
        ]
        self.torch_fp32_multi_output = [
        ]
        self.nn_fp32_multi_output = [
        ]
