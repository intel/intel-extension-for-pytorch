import unittest
import torch
from common_utils import TestCase
from itertools import product
from typing import List, Union
from dataclasses import dataclass
import intel_extension_for_pytorch as ipex

# config alined with vllm
ATOL, RTOL = 6e-2, 6e-2
DTYPES = [torch.float16, torch.bfloat16]
test_params = {
    "hidden_sizes": [2049],
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}
hs_test_params = {
    "hidden_sizes": [
        128,
        256,
        512,
        896,
        1024,
        1152,
        1216,
        1280,
        1536,
        1664,
        2048,
        2240,
        2304,
        2368,
        2432,
        2560,
        2752,
        3072,
        3328,
        3456,
        3584,
        3712,
        4096,
        4480,
        4608,
        4736,
        4864,
        5120,
        5504,
        5632,
        5888,
        6144,
        6400,
        6848,
        6912,
        7168,
        7424,
        8192,
        8960,
        9216,
        9472,
        10240,
        11008,
        11264,
        13824,
        14336,
        14784,
        14848,
        15360,
        18944,
        22016,
        22528,
        24576,
        27392,
        27648,
        29568,
        29696,
        32000,
        32256,
        32512,
        32768,
        33024,
        36864,
        43264,
        49152,
        49408,
        60544,
        60672,
        64000,
        64256,
        102400,
        102656,
        128000,
        128256,
    ],
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}


# test file create based on
# https://github.com/vllm-project/vllm/blob/
# cf069aa8aa38a9003c254f8434a29ec6a3070b08/tests/lora/test_punica_ops.py
# reference copy from
# https://github.com/vllm-project/vllm/blob/
# cf069aa8aa38a9003c254f8434a29ec6a3070b08/vllm/lora/ops/torch_ops/lora_ops.py
def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_expand(inputs, lora_b_weights, output_tensor, exploded_indices, add_inputs)


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    if add_inputs:
        output_tensor[:, : outputs.shape[1]] += outputs[:limit, :]
    else:
        output_tensor[:, : outputs.shape[1]] = outputs[:limit, :]


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_shrink(inputs, lora_a_weights, output_tensor, exploded_indices, scaling)


def bgmv_shrink(
    inputs: torch.Tensor,  # [bs, hidden_size]
    lora_b_weights: torch.Tensor,  # [num_lora, max_rank, hidden_size]
    output_tensor: torch.Tensor,  # [bs, output_size1], output_size1 >= max_rank
    lora_indices_tensor: torch.Tensor,  # [bs]
    scaling: float = 1.0,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    output_tensor[:, : outputs.shape[1]] = scaling * outputs[:]


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor,
        exploded_indices,
        slice_offset,
        slice_size,
        add_inputs,
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    inputs = inputs.to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    if add_inputs:
        output_tensor[:, slice_offset : slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset : slice_offset + slice_size] = outputs[:]


def sgmv_expand_for_nslices(
    nslices: int,
    hidden_size: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: list[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    add_inputs: bool,
) -> None:
    """
    Wrapper around sgmv_expand that handles any nslices.
    """
    if nslices == 1:
        # Verify the torch's sgmv_expand op
        sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights,
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


@dataclass
class PunicaTensors:
    inputs_tensor: torch.Tensor
    lora_weights: Union[torch.Tensor, List[torch.Tensor]]
    our_out_tensor: torch.Tensor
    ref_out_tensor: torch.Tensor
    b_seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    seq_len_tensor: torch.Tensor
    token_lora_mapping: torch.Tensor

    def meta(self):
        """
        Infer max_seq_length and token_nums from the tensors
        and return them.
        """
        max_seq_length = self.seq_len_tensor.max()
        token_nums = self.seq_len_tensor.sum().item()
        if isinstance(max_seq_length, tuple):
            max_seq_length = max_seq_length[0].item()
        else:
            max_seq_length = max_seq_length.item()
        return max_seq_length, token_nums


def generate_data(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)
        lora_weights = torch.rand(
            (lora_nums, max_rank, hidden_size),  # col-major
            dtype=dtype,
        ).to(device)
        # shrink op need atomic_add, so output is initinized by 0
        ref_out_tensor = torch.zeros(
            (total_tokens, max_rank), dtype=dtype, device=inputs_tensor.device
        )
        # NOTE  shrink kernel using torch.float32 as output type
        our_out_tensor = torch.zeros((total_tokens, max_rank), dtype=torch.float32).to(
            device
        )
    else:
        inputs_tensor = torch.rand(
            (total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        lora_weights = torch.rand(
            (lora_nums, hidden_size, max_rank),  # col-major
            dtype=dtype,
        ).to(device)
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        ref_out_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
        ).to(device)
        # Ensure the same input.
        our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    ).to(device)
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]].copy_(
            lora_index
        )
        current_offset += seq_len_tensor[b_id].item()

    return PunicaTensors(
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def generate_data_for_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    nslices,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()

    lora_weights_lst = []
    if op_type == "shrink":

        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)

        for _ in range(nslices):
            if op_type == "shrink":
                lora_weights_lst.append(
                    torch.rand(
                        (lora_nums, max_rank, hidden_size),  # col-major
                        dtype=dtype,
                    ).to(device)
                )
        # NOTE  shrink kernel using torch.float32 as output type
        # shrink op need atomic_add, so output is initinized by 0
        our_out_tensor = torch.zeros(
            (nslices, total_tokens, max_rank),
            dtype=torch.float32,
        ).to(device)
    else:
        inputs_tensor = torch.rand(
            (nslices, total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        for _ in range(nslices):
            lora_weights_lst.append(
                torch.rand(
                    (lora_nums, hidden_size, max_rank),  # col-major
                    dtype=dtype,
                ).to(device)
            )
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        our_out_tensor = torch.rand(
            (total_tokens, hidden_size * nslices), dtype=dtype
        ).to(device)

    # Ensure the same input.
    ref_out_tensor = our_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    )
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]] = (
            lora_index.item()
        )
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def generate_data_for_expand_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    nslices,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    inputs_tensor = torch.rand(
        (total_tokens, max_rank),
        dtype=dtype,
    ).to(device)
    lora_weights_lst = []
    for _ in range(nslices):
        lora_weights_lst.append(
            torch.rand(
                (lora_nums, hidden_size, max_rank),  # col-major
                dtype=dtype,
            ).to(device)
        )
    # expand op needs to complete y+=a@lora_b, so output is
    # initinized randomly
    ref_out_tensor = torch.rand((total_tokens, hidden_size * nslices), dtype=dtype).to(
        device
    )
    # Ensure the same input.
    our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    )
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]] = (
            lora_index.item()
        )
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def sgmv_shrink_for_nslices(
    nslices: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: List[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    scaling: float,
):
    """
    Wrapper around sgmv_shrink that handles any nslices.
    """
    for index in range(nslices):
        sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def check_sgmv_expand(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    add_inputs: bool,
):
    """
    Compare outputs of vllm.sgmv_expand kernel against a reference
    implementation.
    """

    def ipex_sgmv_expand(
        inputs: torch.Tensor,
        lora_b_weights: torch.Tensor,
        output_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        lora_indices_tensor: torch.Tensor,
        batches: int,
        max_seq_length: int,
        token_nums: int,
        add_inputs: bool = False,
    ):
        exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

        ipex.llm.functional.fusions.bgmv_expand(
            inputs, lora_b_weights, output_tensor, exploded_indices, add_inputs
        )

    def ipex_sgmv_expand_slice(
        inputs: torch.Tensor,
        lora_b_weights: torch.Tensor,
        output_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        lora_indices_tensor: torch.Tensor,
        batches: int,
        max_seq_length: int,
        token_nums: int,
        slice_offset: int,
        slice_size: int,
        add_inputs: bool = False,
    ):
        exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

        ipex.llm.functional.fusions.bgmv_expand_slice(
            inputs,
            lora_b_weights,
            output_tensor,
            exploded_indices,
            slice_offset,
            slice_size,
            add_inputs,
        )

    def ipex_sgmv_expand_for_nslices(
        nslices: int,
        hidden_size: int,
        inputs_tensor: torch.Tensor,
        lora_weights_lst: list[torch.Tensor],
        out_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        prompt_lora_mapping: torch.Tensor,
        batches: int,
        max_seq_length: int,
        num_tokens: int,
        add_inputs: bool,
    ) -> None:
        """
        Wrapper around sgmv_expand that handles any nslices.
        """
        if nslices == 1:
            # Verify the torch's sgmv_expand op
            ipex_sgmv_expand(
                inputs_tensor[0],
                lora_weights_lst[0],
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                add_inputs=add_inputs,
            )
        else:
            slice_offset = 0
            for index in range(nslices):
                lora_weights = lora_weights_lst[index]
                ipex_sgmv_expand_slice(
                    inputs_tensor[index],
                    lora_weights,
                    out_tensor,
                    b_seq_start_loc,
                    seq_len_tensor,
                    prompt_lora_mapping,
                    batches,
                    max_seq_length,
                    num_tokens,
                    slice_offset,
                    hidden_size,
                    add_inputs=add_inputs,
                )
                slice_offset += hidden_size

    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
    )

    max_seq_length, token_nums = data.meta()

    data.our_out_tensor = data.our_out_tensor.to(dtype)
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)
    ipex_sgmv_expand_for_nslices(
        nslices,
        hidden_size,
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
        add_inputs=add_inputs,
    )

    sgmv_expand_for_nslices(
        nslices,
        hidden_size,
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
        add_inputs=add_inputs,
    )

    assert torch.allclose(
        data.ref_out_tensor, data.our_out_tensor, atol=ATOL, rtol=RTOL
    )


def check_sgmv_shrink(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    scaling: float,
):
    """
    Compare outputs of vllm.sgmv_shrink kernel against a reference
    implementation.
    """

    def ipex_sgmv_shrink(
        inputs: torch.Tensor,
        lora_a_weights: torch.Tensor,
        output_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        lora_indices_tensor: torch.Tensor,
        batches: int,
        max_seq_length: int,
        token_nums: int,
        scaling: float,
    ):
        exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)
        ipex.llm.functional.fusions.bgmv_shrink(
            inputs, lora_a_weights, output_tensor, exploded_indices, scaling
        )

    def ipex_sgmv_shrink_for_nslices(
        nslices: int,
        inputs_tensor: torch.Tensor,
        lora_weights_lst: list[torch.Tensor],
        out_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        prompt_lora_mapping: torch.Tensor,
        batches: int,
        max_seq_length: int,
        num_tokens: int,
        scaling: float,
    ):
        """
        Wrapper around sgmv_shrink that handles any nslices.
        """
        for index in range(nslices):
            ipex_sgmv_shrink(
                inputs_tensor,
                lora_weights_lst[index],
                out_tensor[index],
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                scaling,
            )

    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "shrink",
        device,
    )
    max_seq_length, token_nums = data.meta()

    data.our_out_tensor = data.our_out_tensor.to(dtype)
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)
    ipex_sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
        scaling,
    )

    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
        scaling,
    )
    assert torch.allclose(
        data.ref_out_tensor, data.our_out_tensor, atol=ATOL, rtol=RTOL
    )


def check_bgmv_expand_slice(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """
    Compare vllm.bgmv_expand_slice against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )

    slice_offset = 0
    data.our_out_tensor = data.our_out_tensor.to(dtype)
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)
    for index in range(nslices):
        ipex.llm.functional.fusions.bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.our_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            hidden_size,
            add_inputs,
        )

        bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.ref_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )

        slice_offset += hidden_size

    assert torch.allclose(
        data.ref_out_tensor, data.our_out_tensor, atol=ATOL, rtol=RTOL
    )


def check_bgmv_expand(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """
    Compare vllm.bgmv_expand against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "expand",
        device,
    )
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)
    bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)

    data.our_out_tensor = data.our_out_tensor.to(dtype)
    ipex.llm.functional.fusions.bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        add_inputs,
    )
    assert torch.allclose(
        data.ref_out_tensor, data.our_out_tensor, atol=ATOL, rtol=RTOL
    )


def check_bgmv_shrink(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    scaling: float,
):
    """
    Compare vllm.bgmv_shrink against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "shrink",
        device,
    )

    data.ref_out_tensor = data.ref_out_tensor.to(dtype)
    bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        scaling,
    )
    data.ref_out_tensor = data.ref_out_tensor.to(dtype)

    data.our_out_tensor = data.our_out_tensor.to(dtype)
    ipex.llm.functional.fusions.bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        scaling,
    )
    assert torch.allclose(
        data.ref_out_tensor, data.our_out_tensor, atol=ATOL, rtol=RTOL
    )


class PunicaTest(TestCase):
    def test_bgmv(self):
        for batch, num_lora, rank, hidden_size, dtype in product(
            test_params["batches"],
            test_params["num_loras"],
            test_params["max_ranks"],
            test_params["hidden_sizes"],
            DTYPES,
        ):
            check_bgmv_shrink(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                dtype=dtype,
                device="cpu",
                scaling=0.5,
            )

            check_bgmv_expand(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                dtype=dtype,
                device="cpu",
                add_inputs=True,
            )

        for batch, num_lora, rank, hidden_size, dtype in product(
            hs_test_params["batches"],
            hs_test_params["num_loras"],
            hs_test_params["max_ranks"],
            hs_test_params["hidden_sizes"],
            DTYPES,
        ):
            check_bgmv_shrink(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                dtype=dtype,
                device="cpu",
                scaling=0.5,
            )

            check_bgmv_expand(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                dtype=dtype,
                device="cpu",
                add_inputs=True,
            )

    def test_bgmv_expand_slice(self):
        for batch, num_lora, rank, hidden_size, nslices, dtype in product(
            test_params["batches"],
            test_params["num_loras"],
            test_params["max_ranks"],
            test_params["hidden_sizes"],
            [2, 3],
            DTYPES,
        ):
            check_bgmv_expand_slice(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                add_inputs=True,
            )

        for batch, num_lora, rank, hidden_size, nslices, dtype in product(
            hs_test_params["batches"],
            hs_test_params["num_loras"],
            hs_test_params["max_ranks"],
            hs_test_params["hidden_sizes"],
            [2, 3],
            DTYPES,
        ):
            check_bgmv_expand_slice(
                batches=batch,
                num_loras=num_lora,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                add_inputs=True,
            )

    def test_sgmv(self):
        for batches, num_loras, rank, hidden_size, nslices, dtype in product(
            test_params["batches"],
            test_params["num_loras"],
            test_params["max_ranks"],
            test_params["hidden_sizes"],
            [1, 2, 3],
            DTYPES,
        ):
            check_sgmv_shrink(
                batches=batches,
                num_loras=num_loras,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                seq_length=128,
                scaling=0.5,
            )
            check_sgmv_expand(
                batches=batches,
                num_loras=num_loras,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                seq_length=128,
                add_inputs=True,
            )

        for batches, num_loras, rank, hidden_size, nslices, dtype in product(
            hs_test_params["batches"],
            hs_test_params["num_loras"],
            hs_test_params["max_ranks"],
            hs_test_params["hidden_sizes"],
            [1, 2, 3],
            DTYPES,
        ):
            check_sgmv_shrink(
                batches=batches,
                num_loras=num_loras,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                seq_length=128,
                scaling=0.5,
            )
            check_sgmv_expand(
                batches=batches,
                num_loras=num_loras,
                rank=rank,
                hidden_size=hidden_size,
                nslices=nslices,
                dtype=dtype,
                device="cpu",
                seq_length=128,
                add_inputs=True,
            )


if __name__ == "__main__":
    test = unittest.main()
