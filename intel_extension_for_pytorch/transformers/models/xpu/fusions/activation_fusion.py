import torch
from typing import Optional


def gelu_quick_xpu(x: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    return torch.ops.torch_ipex.gelu_quick_out(x, out)


def silu_mul_xpu(x: torch.Tensor, y: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.silu_mul(x, y, out)
    return out


def silu_and_mul_xpu(x: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        d = x.size(-1) / 2
        out_shape = x.shape[:-1] + (d,)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return torch.ops.torch_ipex.silu_and_mul(x, out)


def gelu_mul_xpu(
    x: torch.Tensor,
    y: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    approximate: str = "none",
):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.gelu_mul(x, y, out, approximate)
    return out


def gelu_and_mul_xpu(
    x: torch.Tensor, out: Optional[torch.Tensor] = None, approximate: str = "none"
):
    if out is None:
        d = x.size(-1) / 2
        out_shape = x.shape[:-1] + (d,)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return torch.ops.torch_ipex.gelu_and_mul(x, out, approximate)


def add_rms_norm_xpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    out = torch.ops.torch_ipex.add_rms_norm(
        add, x, [x.size(-1)], weight, bias, eps, add_back
    )
    return out


def add_layer_norm_xpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    out = torch.ops.torch_ipex.add_layer_norm(
        add, x, [x.size(-1)], weight, bias, eps, add_back
    )
    return out


def rotary_embedding_batched_xpu(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_nexo_style: bool,
    rotary_dim: int,
    offsets: Optional[torch.Tensor] = None,
):
    if offsets is None:
        torch.ops.torch_ipex.rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_nexo_style, rotary_dim
        )
    else:
        torch.ops.torch_ipex.rotary_embedding_batched(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_nexo_style,
            rotary_dim,
            offsets,
        )
    return query, key


def bgmv_shrink_xpu(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    """
    This function is generally the same as the one in vllm/lora/ops/bgmv_shrink.py

    Args:
        inputs (torch.Tensor): Shape: `[batch_size, hidden_size]`.
        lora_a_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[batch_size, rank]`.
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`. The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        scaling (float):  Scaling factor.

    Semantics:
      for i in range(inputs.size(0)):
        output_tensor[i] +=
            inputs[i] @ lora_a_weights[lora_indices_tensor[i]] * scale
    """
    assert inputs.is_xpu
    assert lora_a_weights.is_xpu
    assert output_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert output_tensor.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,  # vllm used float32 as output_tensor default dtype
    ]
    assert (
        lora_indices_tensor.dtype == torch.int64
    )  # vllm used int64 as lora_indices_tensor dtype

    assert inputs.ndim == 2
    assert output_tensor.ndim == 2
    assert lora_indices_tensor.ndim == 1
    if lora_a_weights.ndim == 4:  # shape:(lora_num,1,rank,hidden_size)
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3  # shape:(lora_num,rank,hidden_size)

    assert inputs.size(1) == lora_a_weights.size(-1)
    assert output_tensor.size(1) == lora_a_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == lora_indices_tensor.size(0)

    torch.ops.torch_ipex.bgmv_shrink(
        output_tensor, inputs, lora_a_weights, lora_indices_tensor, scaling
    )
    return


def bgmv_expand_xpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    """
    This function is generally the same as the one in vllm/lora/ops/bgmv_expand.py

    Args:
        inputs (torch.Tensor): Shape: `[batch_size, hidden_size]`.
        lora_b_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[batch_size, rank]`.
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`. The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.

    Semantics:
      for i in range(inputs.size(0)):
        output_tensor[i] =
            inputs[i] @ lora_b_weights[lora_indices_tensor[i]]
            + (inputs[i] if add_inputs else 0)
    """
    assert inputs.is_xpu
    assert lora_b_weights.is_xpu
    assert output_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_b_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert output_tensor.dtype == lora_b_weights.dtype
    assert (
        lora_indices_tensor.dtype == torch.int64
    )  # vllm used int64 as lora_indices_tensor dtype

    assert inputs.ndim == 2
    assert output_tensor.ndim == 2
    assert lora_indices_tensor.ndim == 1
    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,hidden_size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,hidden_size,rank)

    assert inputs.size(1) == lora_b_weights.size(-1)
    assert output_tensor.size(1) == lora_b_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == lora_indices_tensor.size(0)

    torch.ops.torch_ipex.bgmv_expand_with_slice(
        output_tensor, inputs, lora_b_weights, lora_indices_tensor, 0, add_inputs
    )
    return


def bgmv_expand_slice_xpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    """
    This function is generally the same as the one in vllm/lora/ops/bgmv_expand_slice.py

    Args:
        inputs (torch.Tensor): Shape: `[batch_size, hidden_size]`.
        lora_b_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[batch_size, rank]`.
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`. The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        slice_offset (int): output_tensor's offset
        slice_size (int): current output_tensor's size
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.

    Semantics:
      for i in range(inputs.size(0)):
        output_tensor[i][slice_offset:slice_offset+slice_size] =
            inputs[i] @ lora_b_weights[lora_indices_tensor[i]]
            + (inputs[i] if add_inputs else 0)
    """
    assert inputs.is_xpu
    assert lora_b_weights.is_xpu
    assert output_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_b_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert output_tensor.dtype == lora_b_weights.dtype
    assert (
        lora_indices_tensor.dtype == torch.int64
    )  # vllm used int64 as lora_indices_tensor dtype

    assert inputs.ndim == 2
    assert output_tensor.ndim == 2
    assert lora_indices_tensor.ndim == 1
    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,hidden_size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,hidden_size,rank)

    assert inputs.size(1) == lora_b_weights.size(-1)
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == lora_indices_tensor.size(0)

    assert slice_offset >= 0
    assert slice_offset + slice_size <= output_tensor.size(1)

    torch.ops.torch_ipex.bgmv_expand_with_slice(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        add_inputs,
    )
    return


def sgmv_shrink_xpu(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    scaling: float,
):
    """
    This function is generally the same as the one in vllm/lora/ops/sgmv_shrink.py

    Args:
        inputs (torch.Tensor): Shape: `[num_token, hidden_size]`
        lora_a_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`
        output_tensor (torch.Tensor): Shape: `[num_token, rank]`.
        b_seq_start_loc (torch.Tensor): Shape: `[batch_size]`. The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): Shape: `[batch_size]`. Record the sequence
            length of the sequences in the batch
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`.  The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences
            in the batch
        scaling (float):  Scaling factor.
    """
    assert inputs.is_xpu
    assert lora_a_weights.is_xpu
    assert output_tensor.is_xpu
    assert b_seq_start_loc.is_xpu
    assert seq_len_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert b_seq_start_loc.is_contiguous()
    assert seq_len_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert output_tensor.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,  # vllm used float32 as output_tensor default dtype
    ]
    assert b_seq_start_loc.dtype == torch.int64
    assert seq_len_tensor.dtype == torch.int64
    assert lora_indices_tensor.dtype == torch.int64

    if lora_a_weights.ndim == 4:  # shape:(lora_num,1,rank,hidden_size)
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3  # shape:(lora_num,rank,hidden_size)
    assert inputs.size(1) == lora_a_weights.size(-1)
    assert output_tensor.size(1) == lora_a_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == b_seq_start_loc[-1] + seq_len_tensor[-1]
    assert b_seq_start_loc.size(0) == batches
    assert seq_len_tensor.size(0) == batches
    assert lora_indices_tensor.size(0) == batches

    if torch.xpu.has_xmx() and torch.xpu.has_2d_block_array() and torch.xpu.has_xetla():
        torch.ops.torch_ipex.sgmv_shrink(
            output_tensor,
            inputs,
            lora_a_weights,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            scaling,
        )
    else:  # sgmv use xetla, fallback to bgmv sycl kernel if xetla not suppport
        indice_tensor = torch.empty(
            inputs.size(0), dtype=torch.long, device=lora_indices_tensor.device
        )
        for i in range(batches):
            indice_tensor[
                b_seq_start_loc[i] : b_seq_start_loc[i] + seq_len_tensor[i]
            ] = lora_indices_tensor[i]
        torch.ops.torch_ipex.bgmv_shrink(
            output_tensor, inputs, lora_a_weights, indice_tensor, scaling
        )


def sgmv_expand_xpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    add_inputs: bool = False,
):
    """
    This function is generally the same as the one in vllm/lora/ops/sgmv_expand.py

    Args:
        inputs (torch.Tensor): Shape: `[num_token, hidden_size]`.
        lora_a_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[num_token, rank]`.
        b_seq_start_loc (torch.Tensor): Shape: `[batch_size]`. The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): Shape: `[batch_size]`. Record the sequence
            length of the sequences in the batch
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`.  The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences
            in the batch
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.
    """
    assert inputs.is_xpu
    assert lora_b_weights.is_xpu
    assert output_tensor.is_xpu
    assert b_seq_start_loc.is_xpu
    assert seq_len_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_b_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert b_seq_start_loc.is_contiguous()
    assert seq_len_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert output_tensor.dtype == lora_b_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert output_tensor.dtype in [torch.float16, torch.bfloat16]
    assert b_seq_start_loc.dtype == torch.int64
    assert seq_len_tensor.dtype == torch.int64
    assert lora_indices_tensor.dtype == torch.int64

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,hidden_size, rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,hidden_size,rank)
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert output_tensor.size(1) == lora_b_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == b_seq_start_loc[-1] + seq_len_tensor[-1]
    assert b_seq_start_loc.size(0) == batches
    assert seq_len_tensor.size(0) == batches
    assert lora_indices_tensor.size(0) == batches

    if torch.xpu.has_xmx() and torch.xpu.has_2d_block_array() and torch.xpu.has_xetla():
        torch.ops.torch_ipex.sgmv_expand_with_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            0,
            add_inputs,
        )
    else:  # sgmv use xetla, fallback to bgmv sycl kernel if xetla not suppport
        indice_tensor = torch.empty(
            inputs.size(0), dtype=torch.long, device=lora_indices_tensor.device
        )
        for i in range(batches):
            indice_tensor[
                b_seq_start_loc[i] : b_seq_start_loc[i] + seq_len_tensor[i]
            ] = lora_indices_tensor[i]
        torch.ops.torch_ipex.bgmv_expand_with_slice(
            output_tensor, inputs, lora_b_weights, indice_tensor, 0, add_inputs
        )


def sgmv_expand_slice_xpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    """
    This function is generally the same as the one in vllm/lora/ops/sgmv_expand.py

    Args:
        inputs (torch.Tensor): Shape: `[num_token, hidden_size]`.
        lora_a_weights (torch.Tensor): Shape: `[lora_num, rank, hidden_size]`.
        output_tensor (torch.Tensor): Shape: `[num_token, rank]`.
        b_seq_start_loc (torch.Tensor): Shape: `[batch_size]`. The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): Shape: `[batch_size]`. Record the sequence
            length of the sequences in the batch
        lora_indices_tensor (torch.Tensor): Shape: `[batch_size]`.  The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences
            in the batch
        slice_offset (int): output_tensor's offset
        slice_size (int): current output_tensor's size
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.
    """
    assert inputs.is_xpu
    assert lora_b_weights.is_xpu
    assert output_tensor.is_xpu
    assert b_seq_start_loc.is_xpu
    assert seq_len_tensor.is_xpu
    assert lora_indices_tensor.is_xpu

    assert inputs.is_contiguous()
    assert lora_b_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert b_seq_start_loc.is_contiguous()
    assert seq_len_tensor.is_contiguous()
    assert lora_indices_tensor.is_contiguous()

    assert output_tensor.dtype == lora_b_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert output_tensor.dtype in [torch.float16, torch.bfloat16]
    assert b_seq_start_loc.dtype == torch.int64
    assert seq_len_tensor.dtype == torch.int64
    assert lora_indices_tensor.dtype == torch.int64

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,hidden_size, rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,hidden_size,rank)
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.size(0) == output_tensor.size(0)
    assert inputs.size(0) == b_seq_start_loc[-1] + seq_len_tensor[-1]
    assert b_seq_start_loc.size(0) == batches
    assert seq_len_tensor.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert slice_offset >= 0
    assert slice_offset + slice_size <= output_tensor.size(1)

    if torch.xpu.has_xmx() and torch.xpu.has_2d_block_array() and torch.xpu.has_xetla():
        torch.ops.torch_ipex.sgmv_expand_with_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            slice_offset,
            add_inputs,
        )
    else:  # sgmv use xetla, fallback to bgmv sycl kernel if xetla not suppport
        indice_tensor = torch.empty(
            inputs.size(0), dtype=torch.long, device=lora_indices_tensor.device
        )
        for i in range(batches):
            indice_tensor[
                b_seq_start_loc[i] : b_seq_start_loc[i] + seq_len_tensor[i]
            ] = lora_indices_tensor[i]
        torch.ops.torch_ipex.bgmv_expand_with_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            indice_tensor,
            slice_offset,
            add_inputs,
        )
