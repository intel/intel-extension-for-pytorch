import unittest
import os
import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase


def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    assert bits == 4

    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(
        order_map, dtype=torch.int32, device=int_tensor.device
    ).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(
        0,
        int_tensor.shape[1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    ).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)

    reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor


def dequantize_awq(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        fp16_weight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros
    qweight = awq_qweight
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    infeatures = awq_qweight.shape[0]

    wf = torch.tensor(
        list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    # zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)

    # Dequantize weights.
    scales = awq_scales
    zeros = zeros.contiguous()
    scale_zeros = zeros * scales

    g_idx = torch.tensor(
        [i // group_size for i in range(infeatures)], dtype=torch.int32
    )
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx].bfloat16()

    qdq_weight_T = weight * scale_mat - scale_zeros_mat.bfloat16()

    bf16_weight = qdq_weight_T.T

    return bf16_weight


class TestLLMQuantization(TestCase):
    def test_auto_awq_qlinear(self):
        group_size = 128
        qweights = torch.load(
            os.path.join(os.path.dirname(__file__), "data/awq/qweight.pt"),
            weights_only=True,
        )
        scales = torch.load(
            os.path.join(os.path.dirname(__file__), "data/awq/scales.pt"),
            weights_only=True,
        )
        qzeros = torch.load(
            os.path.join(os.path.dirname(__file__), "data/awq/qzeros.pt"),
            weights_only=True,
        )
        bias = torch.load(
            os.path.join(os.path.dirname(__file__), "data/awq/bias.pt"),
            weights_only=True,
        )
        bf16_weights = dequantize_awq(qweights, qzeros, scales, 4, group_size)
        activation = torch.load(
            os.path.join(os.path.dirname(__file__), "data/awq/act.pt"),
            weights_only=True,
        )
        ref = torch.matmul(activation, bf16_weights.T) + bias
        woqlinear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
            qweights,
            scales,
            qzeros,
            786,
            1024,
            None,
            bias,
            group_size,
            None,
            ipex.llm.quantization.QuantMethod.AWQ_GEMM,
            ipex.llm.quantization.QuantDtype.INT4,
        )
        target = woqlinear(activation)
        self.assertEqual(ref.float(), target.float(), prec=0.1)


if __name__ == "__main__":
    test = unittest.main()
