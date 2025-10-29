import pytest
import torch

import intel_extension_for_pytorch  # noqa

NUM_TOKENS = [1, 3, 256, 2256, 4096]
NUM_EXPERTS = [32, 160, 256, 257]
TOP_KS = [1, 2, 16, 32]
BLOCK_SIZES = [32, 128]


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


torch.manual_seed(0)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
      should be padded to a multiple of block_size,

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    torch.ops.torch_ipex.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    torch.ops.torch_ipex.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """
    Verify that actual_sorted_ids follows the correct expert-level sorting.
    The kerne limplementation may or may not preserve original token order 1Code has comments. Press enter to view.
    in topk_ids in the final sorted_ids however this does not impact quality.
    """
    # Group tokens by expert from the golden implementation
    golden_expert_tokens = _group_tokens_by_expert(
        golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in golden_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"golden={golden_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size
            expert_ids[current_block : current_block + expert_blocks_needed] = expert_id

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]
    return sorted_token_ids, expert_ids, num_tokens_post_pad


@pytest.mark.parametrize("m", NUM_TOKENS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("pad_sorted_ids", [False, True])
def test_moe_align_block_size(
    m: int, topk: int, num_experts: int, block_size: int, pad_sorted_ids: bool
):
    """Test moe_align_block_size without expert mapping"""
    topk_ids = torch.zeros((m, topk), device="xpu", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="xpu")[:topk]
        topk_ids[i] = experts

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            pad_sorted_ids=pad_sorted_ids,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)

    # For sorted_token_ids, verify block-level correctness rather than exact
    # order Tokens within each expert's blocks can be in any order, but expert
    # regions must be correct
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )

    total_tokens = m * topk
    assert (
        actual_num_tokens.item() % block_size == 0
    ), "num_tokens_post_pad should be divisible by block_size"
    assert (
        actual_num_tokens.item() >= total_tokens
    ), "num_tokens_post_pad should be at least total_tokens"
    valid_tokens = actual_sorted_ids[actual_sorted_ids < total_tokens]
    assert len(valid_tokens) == total_tokens, (
        f"Should have exactly {total_tokens} valid tokens," f" got {len(valid_tokens)}"
    )
    assert (actual_expert_ids >= 0).all() and (
        actual_expert_ids < num_experts
    ).all(), "expert_ids should contain valid expert indices"


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("block_size", [64])
def test_moe_align_block_size_with_expert_map(
    m: int, topk: int, num_experts: int, block_size: int
):
    """Test moe_align_block_size with expert mapping (EP scenario)"""
    topk_ids = torch.zeros((m, topk), device="xpu", dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device="xpu")[:topk]
        topk_ids[i] = experts

    expert_map = torch.full((num_experts,), -1, device="xpu", dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            expert_map=expert_map,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )


def test_moe_align_block_size_deterministic():
    m, topk, num_experts, block_size = 128, 2, 32, 64

    torch.manual_seed(42)
    topk_ids = torch.randint(0, num_experts, (m, topk), device="xpu", dtype=torch.int32)

    # expect the results to be reproducible
    results = []
    for _ in range(5):
        sorted_ids, expert_ids, num_tokens = moe_align_block_size(
            topk_ids=topk_ids, block_size=block_size, num_experts=num_experts
        )
        results.append((sorted_ids.clone(), expert_ids.clone(), num_tokens.clone()))

    for i in range(1, len(results)):
        assert torch.equal(
            results[0][0], results[i][0]
        ), "sorted_ids should be deterministic"
        assert torch.equal(
            results[0][1], results[i][1]
        ), "expert_ids should be deterministic"
        assert torch.equal(
            results[0][2], results[i][2]
        ), "num_tokens should be deterministic"


@pytest.mark.parametrize("max_tokens_per_batch", [13, 16, 512])
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
@pytest.mark.parametrize("simulate_empty_batches", [False, True])
def test_batched_moe_align_block_size(
    max_tokens_per_batch: int,
    num_experts: int,
    block_size: int,
    simulate_empty_batches: bool,
):

    def ref_outputs(
        expert_num_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = expert_num_tokens.size(0)

        # Round up so each batch can be split to blocks evenly.
        Msum = round_up(max_tokens_per_batch, block_size) * E
        ref_sorted_ids = torch.empty((Msum,), dtype=torch.int32)
        ref_expert_ids = torch.empty((Msum // block_size,), dtype=torch.int32)
        ref_num_tokens_post_pad = torch.empty((1,), dtype=torch.int32)

        # Initialize
        sentinel = E * max_tokens_per_batch
        ref_sorted_ids.fill_(sentinel)
        ref_expert_ids.fill_(-1)

        # Fill ref_sorted_ids
        i = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            token_offset = expert_id * max_tokens_per_batch
            for j in range(expert_nt):
                ref_sorted_ids[i] = token_offset + j
                i += 1
            # round up i to the next block_size
            i = round_up(i, block_size)

        ref_num_tokens_post_pad[0] = i

        # Fill expert_ids
        nt_ceil_sum = 0
        for expert_id, expert_nt in enumerate(expert_num_tokens):
            expert_ids_offset = nt_ceil_sum // block_size
            ceil_expert_nt = round_up(int(expert_nt.item()), block_size)
            num_blocks = ceil_expert_nt // block_size
            for x in range(num_blocks):
                ref_expert_ids[expert_ids_offset + x] = expert_id
            nt_ceil_sum += ceil_expert_nt

        return (
            ref_sorted_ids.to("xpu"),
            ref_expert_ids.to("xpu"),
            ref_num_tokens_post_pad.to("xpu"),
        )

    # Compute expert_num_tokens
    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch,
        size=(num_experts,),
        device="cpu",
        dtype=torch.int32,
    )
    if simulate_empty_batches:
        # mark half the batches to have 0 tokens
        zero_batches = torch.randperm(num_experts)[: num_experts // 2]
        expert_num_tokens[zero_batches] = 0

    # ref outputs
    ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad = ref_outputs(
        expert_num_tokens
    )

    # outputs
    sorted_ids, expert_ids, num_tokens_post_pad = batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens.to("xpu")
    )

    assert (
        ref_sorted_ids.size() == sorted_ids.size()
    ), f"{ref_sorted_ids.size()} vs {sorted_ids.size()}"
    assert (
        ref_expert_ids.size() == expert_ids.size()
    ), f"{ref_expert_ids.size()} vs {expert_ids.size()}"
    assert (
        ref_num_tokens_post_pad.size() == num_tokens_post_pad.size()
    ), f"{ref_num_tokens_post_pad.size()} vs {num_tokens_post_pad.size()}"

    torch.testing.assert_close(ref_sorted_ids, sorted_ids, atol=0, rtol=0)
    torch.testing.assert_close(ref_expert_ids, expert_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        ref_num_tokens_post_pad, num_tokens_post_pad, atol=0, rtol=0
    )
