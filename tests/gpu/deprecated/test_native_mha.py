import torch
import itertools
import copy
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_self_mha(self):
        def _test_simple(
            use_padding, pad_all, use_nt, need_weights, average_attn_weights
        ):
            embed_dim = 64
            num_heads = 4
            bs = 4
            seq_len = 8

            q = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )

            if use_padding:
                if pad_all:
                    for q_i in q:
                        q_i[-1] = torch.zeros_like(
                            q[0][-1], device=cpu_device, dtype=torch.float32
                        )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    for mask_i in mask:
                        mask_i[-1] = True
                else:
                    q[0][-1] = torch.zeros_like(
                        q[0][-1], device=cpu_device, dtype=torch.float32
                    )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    mask[0][-1] = True

            k = q
            v = q

            qkv = torch.nn.Linear(
                embed_dim, 3 * embed_dim, device=cpu_device, dtype=torch.float32
            )
            proj = torch.nn.Linear(
                embed_dim, embed_dim, device=cpu_device, dtype=torch.float32
            )

            q_xpu = q.to(dpcpp_device)
            k_xpu = k.to(dpcpp_device)
            v_xpu = v.to(dpcpp_device)
            qkv_xpu = copy.deepcopy(qkv).to(dpcpp_device)
            proj_xpu = copy.deepcopy(proj).to(dpcpp_device)

            key_padding_mask = mask if use_padding else None
            res = torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv.weight,
                qkv.bias,
                proj.weight,
                proj.bias,
                key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            res_xpu = torch._native_multi_head_attention(
                q_xpu,
                k_xpu,
                v_xpu,
                embed_dim,
                num_heads,
                qkv_xpu.weight,
                qkv_xpu.bias,
                proj_xpu.weight,
                proj_xpu.bias,
                (
                    key_padding_mask.to(dpcpp_device)
                    if key_padding_mask is not None
                    else None
                ),
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            print(res[0])
            print(res_xpu[0])
            self.assertEqual(res[0], res_xpu[0].to("cpu"))

        use_paddings = [True, False]
        pad_alls = [True, False]
        use_nts = [True, False]
        need_weights = [True, False]
        average_attn_weights = [True, False]

        for (
            use_padding,
            pad_all,
            use_nt,
            need_weight,
            average_attn_weight,
        ) in itertools.product(
            use_paddings, pad_alls, use_nts, need_weights, average_attn_weights
        ):
            _test_simple(use_padding, pad_all, use_nt, need_weight, average_attn_weight)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_encdec_mha(self):
        def _test_simple(
            use_padding, pad_all, use_nt, need_weights, average_attn_weights
        ):
            embed_dim = 64
            num_heads = 4
            bs = 4
            seq_len = 8

            q = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )

            if use_padding:
                if pad_all:
                    for q_i in q:
                        q_i[-1] = torch.zeros_like(
                            q[0][-1], device=cpu_device, dtype=torch.float32
                        )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    for mask_i in mask:
                        mask_i[-1] = True
                else:
                    q[0][-1] = torch.zeros_like(
                        q[0][-1], device=cpu_device, dtype=torch.float32
                    )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    mask[0][-1] = True

            k = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )
            v = k

            qkv = torch.nn.Linear(
                embed_dim, 3 * embed_dim, device=cpu_device, dtype=torch.float32
            )
            proj = torch.nn.Linear(
                embed_dim, embed_dim, device=cpu_device, dtype=torch.float32
            )

            q_xpu = q.to(dpcpp_device)
            k_xpu = k.to(dpcpp_device)
            v_xpu = v.to(dpcpp_device)
            qkv_xpu = copy.deepcopy(qkv).to(dpcpp_device)
            proj_xpu = copy.deepcopy(proj).to(dpcpp_device)

            key_padding_mask = mask if use_padding else None
            res = torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv.weight,
                qkv.bias,
                proj.weight,
                proj.bias,
                key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            res_xpu = torch._native_multi_head_attention(
                q_xpu,
                k_xpu,
                v_xpu,
                embed_dim,
                num_heads,
                qkv_xpu.weight,
                qkv_xpu.bias,
                proj_xpu.weight,
                proj_xpu.bias,
                (
                    key_padding_mask.to(dpcpp_device)
                    if key_padding_mask is not None
                    else None
                ),
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            print(res[0])
            print(res_xpu[0])
            self.assertEqual(res[0], res_xpu[0].to("cpu"))

        use_paddings = [True, False]
        pad_alls = [True, False]
        use_nts = [True, False]
        need_weights = [True, False]
        average_attn_weights = [True, False]

        for (
            use_padding,
            pad_all,
            use_nt,
            need_weight,
            average_attn_weight,
        ) in itertools.product(
            use_paddings, pad_alls, use_nts, need_weights, average_attn_weights
        ):
            _test_simple(use_padding, pad_all, use_nt, need_weight, average_attn_weight)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_generic_mha(self):
        def _test_simple(
            use_padding, pad_all, use_nt, need_weights, average_attn_weights
        ):
            embed_dim = 64
            num_heads = 4
            bs = 4
            seq_len = 8

            q = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )

            if use_padding:
                if pad_all:
                    for q_i in q:
                        q_i[-1] = torch.zeros_like(
                            q[0][-1], device=cpu_device, dtype=torch.float32
                        )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    for mask_i in mask:
                        mask_i[-1] = True
                else:
                    q[0][-1] = torch.zeros_like(
                        q[0][-1], device=cpu_device, dtype=torch.float32
                    )
                    mask = torch.zeros(
                        q.shape[:-1], device=cpu_device, dtype=torch.bool
                    )
                    mask[0][-1] = True

            k = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )
            v = (
                6
                * torch.rand(
                    bs, seq_len, embed_dim, device=cpu_device, dtype=torch.float32
                )
                - 3
            )

            qkv = torch.nn.Linear(
                embed_dim, 3 * embed_dim, device=cpu_device, dtype=torch.float32
            )
            proj = torch.nn.Linear(
                embed_dim, embed_dim, device=cpu_device, dtype=torch.float32
            )

            q_xpu = q.to(dpcpp_device)
            k_xpu = k.to(dpcpp_device)
            v_xpu = v.to(dpcpp_device)
            qkv_xpu = copy.deepcopy(qkv).to(dpcpp_device)
            proj_xpu = copy.deepcopy(proj).to(dpcpp_device)

            key_padding_mask = mask if use_padding else None
            res = torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv.weight,
                qkv.bias,
                proj.weight,
                proj.bias,
                key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            res_xpu = torch._native_multi_head_attention(
                q_xpu,
                k_xpu,
                v_xpu,
                embed_dim,
                num_heads,
                qkv_xpu.weight,
                qkv_xpu.bias,
                proj_xpu.weight,
                proj_xpu.bias,
                (
                    key_padding_mask.to(dpcpp_device)
                    if key_padding_mask is not None
                    else None
                ),
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
            )

            print(res[0])
            print(res_xpu[0])
            self.assertEqual(res[0], res_xpu[0].to("cpu"))

        use_paddings = [True, False]
        pad_alls = [True, False]
        use_nts = [True, False]
        need_weights = [True, False]
        average_attn_weights = [True, False]

        for (
            use_padding,
            pad_all,
            use_nt,
            need_weight,
            average_attn_weight,
        ) in itertools.product(
            use_paddings, pad_alls, use_nts, need_weights, average_attn_weights
        ):
            _test_simple(
                use_padding,
                pad_all,
                use_nt,
                need_weight,
                average_attn_weight,
            )
