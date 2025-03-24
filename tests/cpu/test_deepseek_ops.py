import unittest
import torch
import torch.nn.functional as F
from common_utils import TestCase


def woq_quant_and_pack(weight, group_size):
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqLowpMode,
        WoqActQuantMode,
        quantize_per_channel,
        quantize_per_block,
    )

    dtype = WoqWeightDtype.INT8
    assert group_size == -1, "current fused MOE WOQ only support group size = -1"
    if group_size == -1:
        qweight, scales, zero_points = quantize_per_channel(
            weight, dtype, None, None, False
        )
    else:
        qweight, scales, zero_points = quantize_per_block(
            weight, dtype, group_size, None, None, False
        )

    _op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
        qweight,
        dtype,
        [weight.shape[0], weight.shape[1]],
        scales,
        zero_points,
        None,  # bias
        None,  # g_idx
        None,  # batch size
        group_size,
        WoqLowpMode.BF16,  # lowp-mode
        WoqActQuantMode.NONE,  # act_quant_mode
        False,  # cache_weight_for_large_batch
    )
    # qweight: {N/block_n, K/block_k, block_k, block_n}
    return (
        _op_context.get_weight(),
        _op_context.get_scales(),
        _op_context.get_zero_points(),
        _op_context,
    )


pres = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


def compare(a: torch.Tensor, b: torch.Tensor, debug=False):

    atol = rtol = pres[a.dtype]

    res = torch.allclose(a, b, rtol=rtol, atol=atol)
    assert res, "[Failure] Acc allclose check is failing..."
    max_diff = (a - b).abs().max().item()
    max_index = torch.argmax((a - b).abs()).item()
    a_sum = a.sum().item()
    b_sum = b.sum().item()

    print(
        "Comparing: ",
        res,
        " max_diff = {:.5f}, asum = {:.3f}, bsum = {:.3f}".format(
            max_diff, a_sum, b_sum
        ),
    )


def grouped_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]

    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]

    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# renormalize is True by default
def grouped_topk_r1_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group,
    topk_group,
    e_score_correction_bias,
    routed_scaling_factor,
):

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    scores_for_choice = scores.view(
        hidden_states.size(0), -1
    ) + e_score_correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(hidden_states.size(0), num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            hidden_states.size(0),
            num_expert_group,
            scores.shape[-1] // num_expert_group,  # 32
        )
        .reshape(hidden_states.size(0), -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    _, topk_idx = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)

    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    topk_weight = topk_weight * routed_scaling_factor

    return topk_weight.to(torch.float32), topk_idx.to(torch.int32)


def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_naive_moe(a, w1, w2, score, topk, renormalize):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def ipex_default_woq_moe(a, w1_list, w3_list, w2_list, score, topk, renormalize):
    G = 1
    topk_group = 1
    E = score.size(-1)
    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = grouped_topk_native(
        a, score, topk, renormalize, G, topk_group
    )

    gate_ctx = []
    up_ctx = []
    down_ctx = []
    group_size = -1
    for expert_idx in range(E):
        _, _, _, w1_op_context = woq_quant_and_pack(w1_list[expert_idx], group_size)
        _, _, _, w3_op_context = woq_quant_and_pack(w3_list[expert_idx], group_size)
        _, _, _, w2_op_context = woq_quant_and_pack(w2_list[expert_idx], group_size)
        gate_ctx.append(w1_op_context)
        up_ctx.append(w3_op_context)
        down_ctx.append(w2_op_context)

    final_out = torch.ops.torch_ipex.deepseek_moe_woq(
        a,
        topk_ids.to(torch.int64),
        gate_ctx,
        up_ctx,
        down_ctx,
        topk_weights.to(a.dtype),
        False,
    )
    return final_out


class DeepSeekTester(TestCase):
    # testing fusedMoE modules
    def test_fused_moe(self):
        def fused_moe(a, w1, w2, score, topk, renormalize, is_woq=False):
            G = 1
            topk_group = 1

            B, D = a.shape
            topk_weights = torch.empty(B, topk, dtype=torch.float32)
            topk_ids = torch.empty(B, topk, dtype=torch.int32)
            topk_weights, topk_ids = grouped_topk_native(
                a, score, topk, renormalize, G, topk_group
            )

            if is_woq:
                E = w1.size(0)
                dtype = a.dtype
                w13_qweight_list = []
                w13_scale_list = []
                w13_zp_list = []
                w2_qweight_list = []
                w2_scale_list = []
                w2_zp_list = []
                group_size = -1
                for idx in range(E):
                    w13_qweight, w13_scale, w13_zp, _ = woq_quant_and_pack(
                        w1[idx], group_size
                    )
                    w13_qweight_list.append(w13_qweight)
                    w13_scale_list.append(w13_scale)
                    w13_zp_list.append(w13_zp)
                    w2_qweight, w2_scale, w2_zp, _ = woq_quant_and_pack(
                        w2[idx], group_size
                    )

                    w2_qweight_list.append(w2_qweight)
                    w2_scale_list.append(w2_scale)
                    w2_zp_list.append(w2_zp)
                packed_w1 = torch.stack(w13_qweight_list).detach()
                w13_scale = torch.stack(w13_scale_list).detach().to(dtype)
                w13_zp = torch.stack(w13_zp_list).detach().to(dtype)
                packed_w2 = torch.stack(w2_qweight_list).detach()
                w2_scale = torch.stack(w2_scale_list).detach().to(dtype)
                w2_zp = torch.stack(w2_zp_list).detach().to(dtype)
            else:
                packed_w1 = torch.ops.torch_ipex.convert_weight_packed_bf16(w1)
                packed_w2 = torch.ops.torch_ipex.convert_weight_packed_bf16(w2)
                w13_scale = torch.tensor(0).to(a.dtype)
                w13_zp = torch.tensor(0).to(a.dtype)
                w2_scale = torch.tensor(0).to(a.dtype)
                w2_zp = torch.tensor(0).to(a.dtype)

            inplace = False
            return torch.ops.torch_ipex.fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                inplace,
                True,
                False,
                is_woq,
                w13_scale,
                w13_zp,
                w2_scale,
                w2_zp,
            )

        def run_single_test(m, n, k, e, topk, dtype, renormalize=False, is_woq=False):
            a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
            score = torch.randn((m, e), device="cpu", dtype=dtype)
            w13_list = []
            w1_list = []
            w3_list = []
            w2_list = []
            for _ in range(e):
                w1_ = torch.randn((n, k), device="cpu", dtype=dtype) / 10
                w3_ = torch.randn((n, k), device="cpu", dtype=dtype) / 10
                w2_ = torch.randn((k, n), device="cpu", dtype=dtype) / 10
                w13_concat = torch.concat([w1_, w3_], 0)
                w13_list.append(w13_concat)
                w1_list.append(w1_)
                w3_list.append(w3_)
                w2_list.append(w2_)
            w13 = torch.stack(w13_list).detach().to(dtype)
            w2 = torch.stack(w2_list).detach().to(dtype)
            if is_woq:  # Using WOQ INT8: lowp=BF16, group size= -1
                torch_output = ipex_default_woq_moe(
                    a, w1_list, w3_list, w2_list, score, topk, renormalize
                )
            else:
                torch_output = torch_naive_moe(a, w13, w2, score, topk, renormalize)
            fused_output = fused_moe(
                a, w13, w2, score, topk, renormalize, is_woq=is_woq
            )

            compare(torch_output, fused_output)

        for is_woq in [True, False]:
            run_single_test(
                2, 2048, 2048, 4, 2, torch.bfloat16, renormalize=True, is_woq=is_woq
            )
            run_single_test(
                2, 128, 7168, 4, 2, torch.bfloat16, renormalize=True, is_woq=is_woq
            )
            run_single_test(
                2, 128, 2048, 4, 2, torch.bfloat16, renormalize=True, is_woq=is_woq
            )
            run_single_test(
                2, 2048, 2048, 4, 2, torch.bfloat16, renormalize=True, is_woq=is_woq
            )

    # testing fusedMoE + shardMoE fusion modules
    def test_fuse_moe_with_shared_expert(self):
        def fuse_moe_with_sharedmoe(a, w1, w2, score, topk, renormalize):

            G = 1
            topk_group = 1

            B, D = a.shape
            topk_weights = torch.empty(B, topk, dtype=torch.float32)
            topk_ids = torch.empty(B, topk, dtype=torch.int32)
            topk_weights, topk_ids = grouped_topk_native(
                a, score, topk, renormalize, G, topk_group
            )

            pad_weights = torch.ones(a.size(0), 1)
            pad_ids = torch.full((a.size(0), 1), w1.size(0) - 1).to(torch.int)
            topk_weights = torch.cat(
                (topk_weights.to(torch.float), pad_weights), -1
            ).to(torch.float)
            topk_ids = torch.cat((topk_ids.to(torch.int), pad_ids), -1).to(torch.int)
            packed_w1 = torch.ops.torch_ipex.convert_weight_packed_bf16(w1)
            packed_w2 = torch.ops.torch_ipex.convert_weight_packed_bf16(w2)
            w13_scale = torch.tensor(0).to(a.dtype)
            w13_zp = torch.tensor(0).to(a.dtype)
            w2_scale = torch.tensor(0).to(a.dtype)
            w2_zp = torch.tensor(0).to(a.dtype)
            inplace = False
            return torch.ops.torch_ipex.fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                inplace,
                True,
                False,
                False,
                w13_scale,
                w13_zp,
                w2_scale,
                w2_zp,
            )

        def run_single_test(m, n, k, e, topk, dtype, renormalize=False):

            a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
            score = torch.randn((m, e), device="cpu", dtype=dtype)

            w1_shared = torch.randn((n, k), device="cpu", dtype=dtype) / 10
            w3_shared = torch.randn((n, k), device="cpu", dtype=dtype) / 10
            w13_shard = torch.concat([w1_shared, w3_shared], 0).unsqueeze(0)
            w2_shared = torch.randn((k, n), device="cpu", dtype=dtype) / 10

            w1_ = torch.cat((w1, w13_shard), dim=0)
            w2_ = torch.cat((w2, w2_shared.unsqueeze(0)), dim=0)

            torch_output = torch_naive_moe(a, w1, w2, score, topk, renormalize)
            torch_output = torch_output + torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(a, w1_shared, None))
                * torch.nn.functional.linear(a, w3_shared, None),
                w2_shared,
                None,
            )

            fused_output = fuse_moe_with_sharedmoe(
                a, w1_, w2_, score, topk, renormalize
            )

            compare(torch_output, fused_output)

        run_single_test(2, 2048, 2048, 4, 2, torch.bfloat16, renormalize=True)
        run_single_test(2, 128, 32, 4, 2, torch.bfloat16, renormalize=True)
        run_single_test(2, 128, 32, 4, 2, torch.bfloat16, renormalize=True)
        run_single_test(2, 4096, 1024 + 32, 8, 2, torch.bfloat16, renormalize=True)

    # testing R1/V3 GroupedTopK and also moegate_linear modules
    def test_moegate_r1(self):
        def run_single_test(M, E, G, topk, topk_group, dtype):

            # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
            hidden_states = torch.randn(M, 7168, dtype=dtype)
            gate_weights = torch.randn(E, 7168, dtype=dtype)
            # gating_output = torch.randn(M, E, dtype=dtype) * 2 * M
            e_score_correction_bias = torch.rand(E)
            routed_scaling_factor = torch.tensor(2.5)
            gating_output = torch.nn.functional.linear(
                hidden_states, gate_weights, None
            )
            ref_topk_weights, ref_topk_ids = grouped_topk_r1_native(
                hidden_states.float(),
                gating_output.float(),
                topk,
                G,
                topk_group,
                e_score_correction_bias,
                routed_scaling_factor,
            )

            gate_weights_pack = torch.ops.torch_ipex.convert_weight_packed(
                gate_weights.unsqueeze(0).detach(), True
            )
            logits = torch.ops.torch_ipex.moe_gate_bmm_forward(
                hidden_states, gate_weights_pack, True, E, torch.tensor(0).to(dtype)
            )
            topk_ids, topk_weights = torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                logits,
                topk,
                True,
                G,
                topk_group,
                e_score_correction_bias.unsqueeze(0),
                routed_scaling_factor,
            )
            res = torch.zeros(M, E, dtype=torch.float)
            ref = torch.zeros(M, E, dtype=torch.float)
            res.scatter_(1, topk_ids.long(), topk_weights)
            ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
            compare(res, ref, False)

        for bs in [1, 2, 4, 16]:
            run_single_test(bs, 16, 4, 3, 2, torch.bfloat16)
            run_single_test(bs, 32, 4, 3, 2, torch.bfloat16)
            run_single_test(bs, 32, 4, 3, 2, torch.bfloat16)
            run_single_test(bs, 64, 1, 6, 1, torch.bfloat16)
            run_single_test(bs, 256, 8, 4, 8, torch.bfloat16)


if __name__ == "__main__":
    test = unittest.main()
