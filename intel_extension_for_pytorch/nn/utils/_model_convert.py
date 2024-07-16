import torch
from ._parameter_wrapper import get_shared_parameter_status, patch_state_dict


def replace_customized_linear_with_linear(model):
    if isinstance(model, torch.jit.ScriptModule):
        return
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Linear) and child.__class__.__name__ in [
                "FalconLinear",
                "Linear",
            ]:
                new_m = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=False if child.bias is None else True,
                )
                new_m.weight = child.weight
                if child.bias is not None:
                    new_m.bias = child.bias
                setattr(model, child_name, new_m)
            else:
                replace_customized_linear_with_linear(child)


def replace_dropout_with_identity(model):
    # replace dropout with identity during inference, so that aten::dropout won't be on the JIT graph.
    # This optimization may provide more fusion opportunites on the graph.
    if isinstance(model, torch.jit.ScriptModule):
        return
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                setattr(model, child_name, torch.nn.Identity())
                getattr(model, child_name).p = child.p
            else:
                replace_dropout_with_identity(child)


def convert_model_data_type(model, dtype):
    # convert weights(bias) of model to dtype to reduce dtype reorder
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], "model convert only support bf16 and fp16"

    params_attr = {}
    get_shared_parameter_status(model, params_attr)

    for _, param in model.named_parameters():
        if param is None:
            continue
        if params_attr[param].can_cast_inference(dtype):
            params_attr[param].cast_for_inference(dtype)

    patch_state_dict(model, params_attr, "inference")
    return params_attr, model


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


def unpack_awq_weight(
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
        unpacked awq_qweight (`torch.LongTensor`):
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

    # infeatures = awq_qweight.shape[0]

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

    return weight.contiguous(), zeros.contiguous()


def unpack_gptq_weight(
    gptq_qweight: torch.Tensor,
    gptq_qzeros: torch.Tensor,
    gptq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    assert bits == 4

    qzeros = gptq_qzeros
    qweight = gptq_qweight
    scales = gptq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    wf = torch.tensor(
        list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)

    torch.bitwise_and(weight, (2**bits) - 1, out=weight)

    weight = weight.view(weight.shape[0] * weight.shape[1], weight.shape[2])
    zeros = zeros.view(-1, zeros.shape[-1])

    return weight.contiguous(), zeros.contiguous()


def prepack_awq_weight(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    t, zp = unpack_awq_weight(awq_qweight, awq_qzeros, awq_scales, bits, group_size)
    # # transpose -> [N, K]
    t = t.T.contiguous()
    qweight_ = t[:, 1::2].bitwise_left_shift(4).bitwise_or_(t[:, ::2]).to(torch.uint8)
    scales_ = awq_scales.t().contiguous()
    zp_ = zp.t_().contiguous()

    return qweight_, scales_, zp_


def prepack_gptq_weight(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    t, zp = unpack_gptq_weight(awq_qweight, awq_qzeros, awq_scales, bits, group_size)
    # # transpose -> [N, K]
    t = t.T.contiguous()
    qweight_ = t[:, 1::2].bitwise_left_shift(4).bitwise_or_(t[:, ::2]).to(torch.uint8)
    scales_ = awq_scales.t().contiguous()
    zp_ = zp.t_().contiguous()

    return qweight_, scales_, zp_
