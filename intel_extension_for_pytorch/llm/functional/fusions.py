from typing import Optional, Tuple
import torch
from intel_extension_for_pytorch.llm.modules import (
    RotaryEmbedding,
    RMSNorm,
    FastLayerNorm,
    IndirectAccessKVCacheAttention,
    VarlenAttention,
)


def rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    rotary_dim: int,
    rotary_half: bool,
    position_ids: torch.Tensor = None,
):
    r"""
    Applies RotaryEmbedding (see https://huggingface.co/papers/2104.09864)
                              on the `query ` or `key` before their multi-head attention computation.
    Args:
    - query, key (torch.Tensor) : inputs to be applied with position embeddings, taking shape of
                                  [batch size, sequence length, num_head/num_kv_head, head_dim]
                                  or [num_tokens, num_head/num_kv_head, head_dim] (as well as the output shape).
    - sin/cos (torch.Tensor): [num_tokens, rotary_dim] the sin/cos value tensor generated to be applied on query/key.
    - rotary_ndims (int): the rotary dimension. e.g., 64 for GPTJ. head size for LLama.
    - head_dim (int) : head dim from the input shape.
    - rotary_half (bool) : if False. e.g., GPT-J 6B/ChatGLM, cos/sin is applied to the neighboring 2 elements,
                           so the offset is 1.
                           if True, e.g., for llama, cos/sin is applied to the neighboring rotary_dim elements,
                           so the offset is rotary_dim/2.
    - position_ids (torch.Tensor): Default is None and optional if sin/cos is provided. the according position_ids
                                   for the input. The shape should be [batch size, sequence length].
    Return
    - query, key (torch.Tensor): [batch size, sequence length, num_head/num_kv_head, head_dim]
                                 or [num_tokens, num_head/num_kv_head, head_dim].

    """
    return RotaryEmbedding.apply_function(
        query, key, sin, cos, rotary_dim, rotary_half, position_ids
    )


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float):
    r"""
    Applies RMSnorm on the input (hidden states).
    (see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L76)
    Args:
    - hidden_states(torch.Tensor) : the input tensor to apply RMSNorm.
    - weight (torch.Tensor): the weight to apply RMSnorm.
    - eps (float) : the variance_epsilon to apply RMSnorm.

    """
    return RMSNorm.apply_function(hidden_states, weight, eps)


def fast_layer_norm(
    hidden_states: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
):
    r"""
    Applies PyTorch Layernorm (see https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    on the input (hidden states).
    Args:
    - hidden_states(torch.Tensor) : the input tensor to apply normalization.
    - normalized_shape (int or list) or torch.Size) input shape from an expected input of size.
    - weight (torch.Tensor): the weight to apply normalization.
    - bias (torch.Tensor): an additive bias for normalization.
    - eps (float): a value added to the denominator for numerical stability.

    """

    return FastLayerNorm.apply_function(
        hidden_states, normalized_shape, weight, bias, eps
    )


def indirect_access_kv_cache_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_attn: float,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    head_mask: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[Tuple[torch.Tensor]] = None,
    alibi: Optional[torch.Tensor] = None,
    add_casual_mask: Optional[bool] = True,
    seq_info: Optional[torch.Tensor] = None,
    text_max_length: Optional[int] = 0,
):
    r"""
    kv_cache is used to reduce computation for **Decoder** layer but it also brings memory overheads,
    for example, when using beam search, the kv_cache should be reordered according to the latest beam
    idx and the current key/value should also be concat with kv_cache in the attention layer to get entire
    context to do scale dot product. When the sequence is very long, the memory overhead will be the
    performance bottleneck. This module provides an Indirect Access KV_cache(IAKV), Firstly, IAKV pre-allocates
    buffers(key and value use different buffers) to store all key/value hidden states and beam index information.
    It can use beam index history to decide which beam should be used by a timestamp and this information will
    generate an offset to access the kv_cache buffer.
    Data Format:
    - The shape of the pre-allocated key(value) buffer is [max_seq, beam*batch, head_num, head_size],
      the hidden state of key/value which is the shape of [beam*batch, head_num, head_size] is stored token by token.
      All beam idx information of every timestamp is also stored in a Tensor with the shape of [max_seq, beam*batch].

    forward
    - query (torch.Tensor): Query tensor; shape: (beam*batch, seq_len, head_num, head_dim).
    - key (torch.Tensor): Key tensor; shape: (beam*batch, seq_len, head_num, head_dim).
    - value (torch.Tensor): Value tensor; shape: (beam*batch, seq_len, head_num, head_dim).
    - scale_attn (float):scale used by the attention layer. should be the  sqrt(head_size).
    - layer_past (tuple(torch.Tensor)): tuple(seq_info, key_cache, value_cache, beam-idx).
                                        key_cache: key cache tensor, shape: (max_seq, beam*batch,  head_num, head_dim);
                                        value_cache: value cache tensor, shape: (max_seq, beam*batch,  head_num, head_dim);
                                        beam-idx:  history beam idx, shape:(max_seq, beam*batch);
                                        seq_info: Sequence info tensor, shape:(1, 1, max_seq, max_seq).
    - head_mask (torch.Tensor): Head mask tensor which is not supported by kernel yet.
    - attention_mask(torch.Tensor): Attention mask information.
    - text_max_length (int) : the max length of kv cache to be used for generation (allocate the pre-cache buffer).

    Return:
    - attn_output:  weighted value which is the output of scale dot product. shape (beam*batch, seq_len, head_num, head_size).
    - attn_weights:  The output tensor of the first matmul in scale dot product which is not supported by kernel now.
    - new_layer_past: updated layer_past (seq_info, key_cache, value_cache, beam-idx).

    Notes:
    - How to reorder KV cache when using the format of IndirectAccessKVCacheAttention (e.g., on llama model
      see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1318)
        def _reorder_cache(
            self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        ) -> Tuple[Tuple[torch.Tensor]]:
            if (
                len(past_key_values[0]) == 4 and past_key_values[0][0].shape[-1] == 1
            ):
                for layer_past in past_key_values:
                    layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
                return past_key_values

    """
    return IndirectAccessKVCacheAttention.apply_function(
        query,
        key,
        value,
        scale_attn,
        layer_past,
        head_mask,
        attention_mask,
        alibi,
        add_casual_mask,
        seq_info,
        text_max_length,
    )


def varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    seqlen_q: torch.Tensor,
    seqlen_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    pdropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    return_softmax: bool,
    gen_: torch.Generator,
):
    r"""
    Applies PyTorch scaled_dot_product_attention on the inputs of query, key and value
                              (see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
                              and accept the variant (different) sequence length among the query, key and value.

    Args:
        module init: this module does not have args for module init
        forward:
        - query (torch.Tensor): shape [query_tokens, num_head, head_size], where tokens is total sequence length among batch size.
        - key (torch.Tensor):  shape [key_tokens, num_head, head_size], where tokens is total sequence length among batch size.
        - value (torch.Tensor): shape [value_tokens, num_head, head_size], where tokens is total sequence length among batch size.
        - out (torch.Tensor): buffer to get the results, the shape is the same as query.
        - seqlen_q (torch.Tensor): shape [batch_size + 1], points the current query_tokens among total sequence length.
        - seqlen_k (torch.Tensor): shape [batch_size + 1], points the current key_tokens among total sequence length.
        - max_seqlen_q (int): max/total sequence length of query.
        - max_seqlen_k (int): max/total sequence length of key.
        - pdropout (float): dropout probability; if greater than 0.0, dropout is applied, default is 0.0.
        - softmax_scale (float): scaling factor applied is prior to softmax.
        - is_causal (bool): whether to apply causal attention masking, default is True.

    """
    return VarlenAttention.apply_function(
        query,
        key,
        value,
        out,
        seqlen_q,
        seqlen_k,
        max_seqlen_q,
        max_seqlen_k,
        pdropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        return_softmax,
        gen_,
    )
