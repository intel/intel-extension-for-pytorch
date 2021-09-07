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
            ("log_softmax", pointwise0_fp32 + (0,)),
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
            ("avg_pool1d", dummy_bf16[2], {"kernel_size": 3, "stride": 1}),
            ("binary_cross_entropy_with_logits", mat0_bf16 + (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
            ("instance_norm", dummy_bf16[2], {"weight": None, "bias": None, "running_mean": torch.rand((n), dtype=torch.float32),
                                              "running_var": torch.rand((n), dtype=torch.float32), "use_input_stats": False,
                                              "momentum": 0.1, "eps": 1e-5, "cudnn_enabled": False}),
            ("grid_sampler", (torch.randn((2, 3, 33, 22), dtype=torch.bfloat16, device=dev),
                              torch.randn((2, 22, 11, 2), dtype=torch.bfloat16, device=dev),
                              0, 0, False)),
            ("fmod", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1.5)),
            ("prod", torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)),
            ("quantile", (torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16), torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16))),
            ("nanquantile", (torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16), torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16))),
            ("stft", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1, 1)),
            ("cdist", (dummy_bf16[1][0], dummy_bf16[1][0])),
            ("cross", (torch.randn(4, 3, dtype=torch.bfloat16), torch.randn(4, 3, dtype=torch.bfloat16))),
            ("cumprod", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("cumsum", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("diag", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("diagflat", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("histc", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("logcumsumexp", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("tril", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("triu", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), 1)),
            ("vander", (torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16))),
            ("dot", (dummy_bf16[0][0], dummy_bf16[0][0])),
            ("inverse", mat2_bf16),
            ("pinverse", mat2_bf16),
            ("max_pool3d", dummy_bf16[3], {"kernel_size": (3, 3, 3), "stride": (1, 1, 1)}),
            ("selu", dummy_bf16[2]),
            ("celu", dummy_bf16[2]),
            ("group_norm", torch.randn(1, 6, 10, 10).to(torch.bfloat16), {"num_groups": 1}),
            ("conv_tbc", (torch.randn(2, 1, 8).to(torch.bfloat16), torch.randn(3, 8, 8).to(torch.bfloat16), dummy_bf16[0][0])),
        ]
        self.nn_bf16 = [
            ("linear", mat0_fp32 + mat1_fp32),
        ]
        self.fft_fp32 = [
            ("fft_fft", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_ifft", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_fft2", torch.randn(1, 4).to(torch.bfloat16), {"dim": -1}),
            ("fft_ifft2", torch.randn(1, 4).to(torch.bfloat16), {"dim": -1}),
            ("fft_fftn", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_ifftn", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_rfft", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_irfft", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_rfft2", torch.randn(1, 4).to(torch.bfloat16), {"dim": -1}),
            ("fft_irfft2", torch.randn(1, 4).to(torch.bfloat16), {"dim": -1}),
            ("fft_rfftn", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_irfftn", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_hfft", torch.randn(1, 4).to(torch.bfloat16)),
            ("fft_ihfft", torch.randn(1, 4).to(torch.bfloat16)),
        ]
        self.special_fp32 = [
        ]
        self.linalg_fp32 = [
            ("linalg_matrix_norm", dummy_bf16[2]),
            ("linalg_cond", dummy_bf16[2]),
            ("linalg_matrix_rank", dummy_bf16[2]),
            ("linalg_solve", dummy_bf16[2], {"other": dummy_bf16[2][0]}),
            ("linalg_cholesky", torch.mm(dummy_bf16[1][0], dummy_bf16[1][0].t()).reshape(1, 8, 8)),
            ("linalg_svdvals", dummy_bf16[2]),
            ("linalg_eigvals", dummy_bf16[2]),
            ("linalg_eigvalsh", dummy_bf16[2]),
            ("linalg_inv", dummy_bf16[2]),
            ("linalg_householder_product", (dummy_bf16[1][0], dummy_bf16[0][0])),
            ("linalg_tensorinv", dummy_bf16[1], {"ind": 1}),
            ("linalg_tensorsolve", (torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4)).to(torch.bfloat16), torch.randn(2 * 3, 4).to(torch.bfloat16))),
            ("linalg_qr", dummy_bf16[1]),
            ("linalg_cholesky_ex", dummy_bf16[1]),
            ("linalg_svd", dummy_bf16[1]),
            ("linalg_eig", dummy_bf16[1]),
            ("linalg_eigh", dummy_bf16[1]),
            ("linalg_lstsq", (dummy_bf16[1][0], dummy_bf16[1][0])),
        ]
        self.nn_fp32 = [
            ("avg_pool2d", dummy_bf16[2], {"kernel_size": (3, 2), "stride": (1, 1)}),
            ("avg_pool3d", dummy_bf16[3], {"kernel_size": (3, 3, 3), "stride": (1, 1, 1)}),
            ("binary_cross_entropy", (torch.rand((n, n), device=dev, dtype=torch.bfloat16),) +
                                     (torch.rand((n, n), device=dev, dtype=torch.bfloat16),)),
            ("reflection_pad1d", dummy_bf16[2], {"padding": (3, 3)}),
            ("adaptive_avg_pool3d", torch.randn(1, 64, 10, 9, 8).to(torch.bfloat16), {"output_size": 7}),
            ("reflection_pad1d", torch.arange(8, dtype=torch.float).reshape(1, 2, 4).to(torch.bfloat16), {"padding": 2}),
            ("reflection_pad2d", torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).to(torch.bfloat16), {"padding": 2}),
            ("replication_pad1d", torch.arange(8, dtype=torch.float).reshape(1, 2, 4).to(torch.bfloat16), {"padding": 2}),
            ("replication_pad2d", torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).to(torch.bfloat16), {"padding": 2}),
            ("replication_pad3d", torch.arange(1 * 3 * 8 * 320 * 480, dtype=torch.float).reshape(1, 3, 8, 320, 480).to(torch.bfloat16), {"padding": 3}),
            ("elu", dummy_bf16[2]),
            ("hardshrink", dummy_bf16[2]),
            ("hardsigmoid", dummy_bf16[2]),
            ("hardswish", dummy_bf16[2]),
            ("log_sigmoid", dummy_bf16[2]),
            ("prelu", torch.randn(1, 3, 2, 2).to(torch.bfloat16), {"weight": torch.randn(2)}),
            ("softplus", dummy_bf16[2]),
            ("softshrink", dummy_bf16[2]),
            ("mse_loss", (torch.randn(3, 5, requires_grad=True).to(torch.bfloat16), torch.randn(3, 5).to(torch.bfloat16))),
        ]
        self.torch_need_autocast_promote = [
            ("cat", (pointwise0_bf16 + pointwise1_fp32,)),
            ("stack", (pointwise0_bf16 + pointwise1_fp32,)),
            ("index_copy", (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16), 0, torch.tensor([0, 1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))),
        ]
        self.blacklist_non_float_output_pass_test = [
            ("multinomial", (torch.tensor([0, 10, 3, 0], dtype=torch.bfloat16), 4, True)),
            ("searchsorted", (torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.bfloat16), torch.tensor([[3, 6, 9], [3, 6, 9]]))),
            ("vdot", (torch.tensor([2, 3], dtype=torch.bfloat16), torch.tensor([2, 1], dtype=torch.bfloat16))),
        ]
        self.torch_fp32_multi_output = [
            ("cummax", (torch.randn(10).to(torch.bfloat16), 0)),
            ("cummin", (torch.randn(10).to(torch.bfloat16), 0)),
            ("eig", (torch.randn(10, 10).to(torch.bfloat16), True)),
            ("geqrf", (torch.randn(10, 10).to(torch.bfloat16), )),
            ("lstsq", (torch.randn(10, 10).to(torch.bfloat16), torch.randn(10, 10).to(torch.bfloat16))),
            ("_lu_with_info", (torch.randn(10, 10).to(torch.bfloat16), True)),
            ("qr", (torch.randn(10, 10).to(torch.bfloat16), True)),
            ("solve", (torch.randn(10, 10).to(torch.bfloat16), torch.randn(10, 10).to(torch.bfloat16))),
            ("svd", (torch.randn(10, 10).to(torch.bfloat16), True)),
            ("symeig", (torch.randn(10, 10).to(torch.bfloat16), True)),
            ("triangular_solve", (torch.randn(10, 10).to(torch.bfloat16), torch.randn(10, 10).to(torch.bfloat16))),
            ("adaptive_max_pool1d", (torch.randn(100, 100, 100).to(torch.bfloat16), 13)),
            ("adaptive_max_pool2d", (torch.randn(100, 100, 100).to(torch.bfloat16), (13, 13))),
            ("adaptive_max_pool3d", (torch.randn(100, 100, 100, 100).to(torch.bfloat16), (13, 13, 13))),
        ]
        self.nn_fp32_multi_output = [
            ("fractional_max_pool2d", (torch.randn(100, 100, 100).to(torch.bfloat16), 2, (13, 12), torch.randn(10, 10, 10))),
            ("fractional_max_pool3d", (torch.randn(100, 100, 100, 100).to(torch.bfloat16), 2, (13, 12, 1), torch.randn(10, 10, 10))),
        ]
