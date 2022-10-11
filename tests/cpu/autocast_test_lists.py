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

        pointwise0_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        pointwise1_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        pointwise2_fp16 = (torch.randn(n, dtype=torch.float16, device=dev),)
        mat0_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)
        mat1_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)
        mat2_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)
        mat3_fp16 = (torch.randn((n, n), dtype=torch.float16, device=dev),)

        dummy_dimsets = ((n,), (n, n), (n, n, n), (n, n, n, n), (n, n, n, n, n))

        dummy_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),)
                      for dimset in dummy_dimsets]
        dummy_fp16 = [(torch.randn(dimset, dtype=torch.float16, device=dev),)
                      for dimset in dummy_dimsets]

        dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
        conv_args_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),
                           torch.randn(dimset, dtype=torch.bfloat16, device=dev))
                          for dimset in dimsets]
        conv_args_fp16 = [(torch.randn(dimset, dtype=torch.float16, device=dev),
                           torch.randn(dimset, dtype=torch.float16, device=dev))
                          for dimset in dimsets]
        conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                           torch.randn(dimset, dtype=torch.float32, device=dev))
                          for dimset in dimsets]

        bias_fp32 = (torch.randn((n,), dtype=torch.float32, device=dev),)
        bias_fp16 = (torch.randn((n,), dtype=torch.float16, device=dev),)
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
        self.torch_expect_builtin_promote_bf16 = [
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
        self.torch_expect_builtin_promote_fp16 = [
            ("eq", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ge", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("gt", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("le", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("lt", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("ne", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("add", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("div", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("mul", pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]
        self.methods_expect_builtin_promote_bf16 = [
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
        self.methods_expect_builtin_promote_fp16 = [
            ("__eq__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ge__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__gt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__le__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__lt__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__ne__", pointwise0_fp32 + pointwise1_fp16, torch.bool),
            ("__add__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__div__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
            ("__mul__", pointwise0_fp32 + pointwise1_fp16, torch.float32),
        ]
        # The remaining lists organize ops that autocast treats explicitly
        self.fft_fp32 = [
        ]
        self.special_fp32 = [
        ]
        self.linalg_fp32 = [
            ("linalg_matrix_rank", dummy_bf16[2]),
        ]
        self.blacklist_non_float_output_pass_test = [
        ]
        # The remaining lists organize ops that autocast treats explicitly for bf16.
        self.torch_need_autocast_promote_bf16 = [
            ("cat", (pointwise0_bf16 + pointwise1_fp32,)),
            ("stack", (pointwise0_bf16 + pointwise1_fp32,)),
            ("index_copy", (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16), 0, torch.tensor([0, 1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))),
        ]
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
            ("group_norm", (torch.randn((4, 8, 10, 10), device=dev, dtype=torch.float32),
                            4, torch.randn(8, device=dev, dtype=torch.float32),
                            torch.randn(8, device=dev, dtype=torch.float32), 1e-5, True)),
        ]
        self.torch_bf16_multi_output = [
            ("_native_multi_head_attention", (torch.randn((1, 1, 768), device=dev, dtype=torch.float32),
                                              torch.randn((1, 1, 768), device=dev, dtype=torch.float32),
                                              torch.randn((1, 1, 768), device=dev, dtype=torch.float32),
                                              768, 12, torch.randn((2304, 768), device=dev, dtype=torch.float32),
                                              torch.randn((2304), device=dev, dtype=torch.float32),
                                              torch.randn((768, 768), device=dev, dtype=torch.float32),
                                              torch.randn((768), device=dev, dtype=torch.float32),
                                              None, False, True, 1)),
            ("_native_multi_head_attention", (torch.randn((1, 2, 768), device=dev, dtype=torch.float32),
                                              torch.randn((1, 2, 768), device=dev, dtype=torch.float32),
                                              torch.randn((1, 2, 768), device=dev, dtype=torch.float32),
                                              768, 12, torch.randn((2304, 768), device=dev, dtype=torch.float32),
                                              torch.randn((2304), device=dev, dtype=torch.float32),
                                              torch.randn((768, 768), device=dev, dtype=torch.float32),
                                              torch.randn((768), device=dev, dtype=torch.float32),
                                              torch.Tensor([[False, True]]), False, True, 1)),
            ("_transform_bias_rescale_qkv", (torch.randn((1, 96, 1536), device=dev, dtype=torch.float32),
                                             torch.randn((1536), device=dev, dtype=torch.float32), 8)),
        ]
        self.torch_bf16_fp32 = [
        ]
        self.nn_bf16 = [
            ("linear", mat0_fp32 + mat1_fp32),
        ]
        self.nn_bf16_fp32 = [
        ]
        self.torch_bf16_fp32_multi_output = [
        ]
        self.nn_bf16_fp32_multi_output = [
        ]

        # The remaining lists organize ops that autocast treats explicitly for fp16.
        self.torch_need_autocast_promote_fp16 = [
            ("cat", (pointwise0_fp16 + pointwise1_fp32,)),
            ("stack", (pointwise0_fp16 + pointwise1_fp32,)),
            ("index_copy", (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16), 0, torch.tensor([0, 1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))),
        ]
        self.torch_fp16 = [
        ]
        self.torch_fp16_fp32_multi_output = [
            ("_native_multi_head_attention", (torch.randn((1, 1, 768), device=dev, dtype=torch.float16),
                                              torch.randn((1, 1, 768), device=dev, dtype=torch.float16),
                                              torch.randn((1, 1, 768), device=dev, dtype=torch.float16),
                                              768, 12, torch.randn((2304, 768), device=dev, dtype=torch.float16),
                                              torch.randn((2304), device=dev, dtype=torch.float16),
                                              torch.randn((768, 768), device=dev, dtype=torch.float16),
                                              torch.randn((768), device=dev, dtype=torch.float16),
                                              None, False, True)),
            ("_native_multi_head_attention", (torch.randn((1, 2, 768), device=dev, dtype=torch.float16),
                                              torch.randn((1, 2, 768), device=dev, dtype=torch.float16),
                                              torch.randn((1, 2, 768), device=dev, dtype=torch.float16),
                                              768, 12, torch.randn((2304, 768), device=dev, dtype=torch.float16),
                                              torch.randn((2304), device=dev, dtype=torch.float16),
                                              torch.randn((768, 768), device=dev, dtype=torch.float16),
                                              torch.randn((768), device=dev, dtype=torch.float16),
                                              torch.Tensor([[False, True]]), False, True, 1)),
            ("_transform_bias_rescale_qkv", (torch.randn((1, 96, 1536), device=dev, dtype=torch.float16),
                                             torch.randn((1536), device=dev, dtype=torch.float16), 8)),
        ]
        self.nn_fp16 = [
        ]
        self.torch_fp16_fp32 = [
            ("conv1d", conv_args_fp16[0]),
            ("conv2d", conv_args_fp16[1]),
            ("conv3d", conv_args_fp16[2]),
            ("conv_tbc", conv_args_fp16[0] + bias_fp16),
            ("_convolution", conv_args_fp32[1] + bias_fp16 + ((1, 1), (0, 0), (1, 1), False,
                                                              (0, 0), 1, False, True, True)),
            ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float16),
                     torch.randn((n, n, n), device=dev, dtype=torch.float16))),
            ("mm", mat0_fp16 + mat1_fp16),
            ("addmm", mat1_fp16 + mat2_fp16 + mat3_fp16),
            ("addbmm", mat0_fp16 + (torch.randn((n, n, n), device=dev, dtype=torch.float16),
                                    torch.randn((n, n, n), device=dev, dtype=torch.float16))),
            ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float16),
                         torch.randn((n, n, n), device=dev, dtype=torch.float16),
                         torch.randn((n, n, n), device=dev, dtype=torch.float16))),
            ("matmul", mat0_fp16 + mat1_fp16),
            ("conv_transpose1d", conv_args_fp16[0]),
            ("conv_transpose2d", conv_args_fp16[1]),
            ("conv_transpose3d", conv_args_fp16[2]),
            ("group_norm", (torch.randn((4, 8, 10, 10), device=dev, dtype=torch.float16),
                            4, torch.randn(8, device=dev, dtype=torch.float16),
                            torch.randn(8, device=dev, dtype=torch.float16), 1e-5, True)),
            ("batch_norm", dummy_fp16[2], {"weight": None, "bias": None, "running_mean": torch.rand((n), dtype=torch.float32),
                                           "running_var": torch.rand((n), dtype=torch.float32), "training": False,
                                           "momentum": 0.1, "eps": 1e-5, "cudnn_enabled": False}),
            ("avg_pool1d", dummy_fp16[2], {"kernel_size": 3, "stride": 1}),
            ("max_pool1d", (torch.randn(10, 10, 10).to(torch.float16), 3, 2, 0, 1, False)),
            ("max_pool2d", (torch.randn(10, 10, 10, 10).to(torch.float16), 3, 2, 0, 1, False)),
            ("max_pool3d", (torch.randn(10, 10, 10, 10, 10).to(torch.float16), 3, 2, 0, 1, False)),
            ("layer_norm", pointwise0_fp16 + ((pointwise0_fp16[0].numel(),),)),
            ("dropout", mat0_fp16 + (0.5,) + (False,)),
            ("tanh", mat0_fp16),

        ]
        self.nn_fp16_fp32 = [
            ("mish", mat0_fp16),
            ("linear", mat0_fp16 + mat1_fp16),
            ("avg_pool2d", dummy_fp16[2], {"kernel_size": (3, 2), "stride": (1, 1)}),
            ("avg_pool3d", dummy_fp16[3], {"kernel_size": (3, 3, 3), "stride": (1, 1, 1)}),
            ("gelu", mat0_fp16),
        ]
