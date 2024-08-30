import torch
import torch.nn as nn
from typing import Optional, Tuple
from .utils import IPEXRuntimeCustomOps, IPEXCustomOpType


class RotaryEmbedding(nn.Module):
    r"""
    [module init and forward] Applies RotaryEmbedding (see https://huggingface.co/papers/2104.09864)
    on the ``query`` or ``key`` before their multi-head attention computation.

    `module init`

    Args:
        max_position_embeddings (int): size (max) of the position embeddings.
        pos_embd_dim  (int):  dimension of the position embeddings.
        base (int) : Default: 10000. Base to generate the frequency of position embeddings.
        backbone (str): Default: None. The exact transformers model backbone
            (e.g., "GPTJForCausalLM", get from model.config.architectures[0],
            see https://huggingface.co/EleutherAI/gpt-j-6b/blob/main/config.json#L4).
        extra_rope_config (dict): like phi-3 model, it uses original_max_position_embeddings,
            long_factor and short_factor, see details:
            https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json#L23.

    `forward()`

    Args:
        input (torch.Tensor) : input to be applied with position embeddings,
            taking shape of [batch size, sequence length, num_head/num_kv_head, head_dim]
            (as well as the output shape).
        position_ids (torch.Tensor): the according position_ids for the input.
            The shape should be [batch size, sequence length. In some cases,
            there is only one element which the past_kv_length, and position id
            can be constructed by past_kv_length + current_position.
        num_head (int) : head num from the input shape.
        head_dim (int) : head dim from the input shape.
        offset (int) : the offset value. e.g., GPT-J 6B/ChatGLM, cos/sin is applied to the neighboring 2 elements,
            so the offset is 1. For llama, cos/sin is applied to the neighboring rotary_dim elements,
            so the offset is rotary_dim/2.
        rotary_ndims (int): the rotary dimension. e.g., 64 for GPTJ. head size for LLama.

    Examples:
        >>> # module init:
        >>> rope_module = ipex.llm.modules.RotaryEmbedding(2048, 64, base=10000, backbone="GPTJForCausalLM")
        >>> # forward:
        >>> query = torch.randn(1, 32, 16, 256)
        >>> position_ids  = torch.arange(32).unsqueeze(0)
        >>> query_rotery = rope_module(query, position_ids, 16, 256, 1, 64)

    [Direct function call] This module also provides a `.apply_function` function call
    to be used on query and key at the same time without initializing the module
    (assume rotary embedding sin/cos values are provided).

    `apply_function()`

    Args:
        query, key (torch.Tensor) : inputs to be applied with position embeddings, taking shape of
            [batch size, sequence length, num_head/num_kv_head, head_dim]
            or [num_tokens, num_head/num_kv_head, head_dim] (as well as the output shape).
        sin/cos (torch.Tensor): [num_tokens, rotary_dim] the sin/cos value tensor generated to be applied on query/key.
        rotary_ndims (int): the rotary dimension. e.g., 64 for GPTJ. head size for LLama.
        head_dim (int) : head dim from the input shape.
        rotary_half (bool) : if False. e.g., GPT-J 6B/ChatGLM, cos/sin is applied to the neighboring 2 elements,
            so the offset is 1.
            if True, e.g., for llama, cos/sin is applied to the neighboring rotary_dim elements,
            so the offset is rotary_dim/2.
        position_ids (torch.Tensor): Default is None and optional if sin/cos is provided. the according position_ids
            for the input. The shape should be [batch size, sequence length].

    Return:
        query, key (torch.Tensor): [batch size, sequence length, num_head/num_kv_head, head_dim]
        or [num_tokens, num_head/num_kv_head, head_dim].

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(
        self,
        max_position_embeddings: int,
        pos_embd_dim: int,
        base=10000,
        backbone: str = None,
        extra_rope_config: dict = None,
    ):
        super().__init__()
        self.model_backbone = backbone
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_dim = pos_embd_dim
        self.base = base
        self.extra_rope_config = extra_rope_config

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        num_head: int,
        head_dim: int,
        offset: int,
        rotary_ndims: int,
        seq_len: Optional[int] = None,
        num_concats: Optional[int] = None,
    ):
        # Usage 1 (concat query, key, value as input):
        # concat_qkv (in shape) : [batch, seqlen, hidden_size*3]
        # query, key, value (out shape) : [batch, seqlen, num_head/num_kv_head, head_dim]
        # sin, cos: [seqlen, rotary_dim]
        # position_ids: [batch, seqlen]

        # Usage 2 (query, key as input):
        # query/key (in/out shape) : [batch, seqlen, num_head/num_kv_head, head_dim]
        # sin, cos: [seqlen, rotary_dim]
        # position_ids: [batch, seqlen]

        runtime_module = self.runtime_ops.get_module_from_device(
            x.device.type,
            IPEXCustomOpType.ROPE,
            True,
            self.max_position_embeddings,
            self.pos_embd_dim,
            self.base,
            self.model_backbone,
            self.extra_rope_config,
        )
        return runtime_module(
            x,
            position_ids,
            num_head,
            head_dim,
            offset,
            rotary_ndims,
            seq_len,
            num_concats,
        )

    @classmethod
    def apply_function(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        rotary_dim: int,
        rotary_half: bool,
        position_ids: torch.Tensor = None,
    ):
        # query, key (in/out shape) torch.Tensor :
        #    4D: [batch, seqlen, num_head/num_kv_head, head_dim]
        #    3D: [num_tokens, num_head/num_kv_head, head_dim]
        # sin, cos: torch.Tensor [num_tokens, rotary_dim]
        # position_ids (optional): torch.Tensor [batch, seqlen]

        runtime_module = cls.runtime_ops.get_module_from_device(
            query.device.type, IPEXCustomOpType.ROPE, False
        )
        query, key = runtime_module.rotary_embedding(
            query, key, sin, cos, rotary_dim, rotary_half, position_ids
        )
        return query, key


class FastLayerNorm(nn.Module):
    r"""
    [module init and forward] Applies PyTorch Layernorm (see https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    on the input (hidden states).

    `module init`

    Args:
        normalized_shape ((int or list) or torch.Size) input shape from an expected input of size.
        eps (float): a value added to the denominator for numerical stability.
        weight (torch.Tensor): the weight of Layernorm to apply normalization.
        bias (torch.Tensor): an additive bias for normalization.

    `forward()`

    Args:
        hidden_states (torch.Tensor) : input to be applied Layernorm, usually taking shape of
            [batch size, sequence length, hidden_size] (as well as the output shape).

    Examples:
        >>> # module init:
        >>> layernorm = torch.nn.LayerNorm(4096)
        >>> layernorm_module = ipex.llm.modules.FastLayerNorm(4096, eps=1e-05, weight=layernorm.weight, bias=layernorm.bias)
        >>> # forward:
        >>> input = torch.randn(1, 32, 4096)
        >>> result = layernorm_module(input)

    [Direct function call] This module also provides a `.apply_function` function call to apply fast layernorm
    without initializing the module.

    `apply_function()`

    Args:
        hidden_states(torch.Tensor) : the input tensor to apply normalization.
        normalized_shape (int or list) or torch.Size) input shape from an expected input of size.
        weight (torch.Tensor): the weight to apply normalization.
        bias (torch.Tensor): an additive bias for normalization.
        eps (float): a value added to the denominator for numerical stability.

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = weight
        self.bias = bias

    @classmethod
    def apply_function(cls, hidden_states, normalized_shape, weight, bias, eps):
        return cls.runtime_ops.get_module_from_device(
            hidden_states.device.type, IPEXCustomOpType.FAST_LAYERNORM, False
        ).apply_function(hidden_states, normalized_shape, weight, bias, eps)

    def forward(self, hidden_states: torch.Tensor):
        runtime_module = self.runtime_ops.get_module_from_device(
            hidden_states.device.type,
            IPEXCustomOpType.FAST_LAYERNORM,
            True,
            self.normalized_shape,
            self.eps,
            self.weight,
            self.bias,
        )
        return runtime_module(hidden_states)


class RMSNorm(nn.Module):
    r"""
    [module init and forward] Applies RMSnorm on the input (hidden states).
    (see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L76)

    `module init`

    Args:
        hidden_size (int) : the size of the hidden states.
        eps (float) : the variance_epsilon to apply RMSnorm, default using 1e-6.
        weight (torch.Tensor): the weight to apply RMSnorm, default None
            and will use `torch.ones(hidden_size)`.

    `forward()`

    Args:
        hidden_states (torch.Tensor) : input to be applied RMSnorm, usually taking shape of
            [batch size, sequence length, hidden_size] (as well as the output shape).

    Examples:
        >>> # module init:
        >>> rmsnorm_module = ipex.llm.modules.RMSNorm(4096)
        >>> # forward:
        >>> input = torch.randn(1, 32, 4096)
        >>> result = rmsnorm_module(input)

    [Direct function call] This module also provides a `.apply_function` function
    call to apply RMSNorm without initializing the module.

    `apply_function()`

    Args:
        hidden_states(torch.Tensor) : the input tensor to apply RMSNorm.
        weight (torch.Tensor): the weight to apply RMSnorm.
        eps (float) : the variance_epsilon to apply RMSnorm.

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, weight: torch.Tensor = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = (
            weight if weight is not None else nn.Parameter(torch.ones(hidden_size))
        )

    @classmethod
    def apply_function(cls, hidden_states, weight, eps):
        return cls.runtime_ops.get_module_from_device(
            hidden_states.device.type, IPEXCustomOpType.RMS_NORM, False
        ).apply_function(hidden_states, weight, eps)

    def forward(self, x: torch.Tensor):
        runtime_module = self.runtime_ops.get_module_from_device(
            x.device.type, IPEXCustomOpType.RMS_NORM, True, self
        )
        return runtime_module(x)


class VarlenAttention(nn.Module):
    r"""
    [module init and forward] Applies PyTorch scaled_dot_product_attention on the inputs of query, key and value
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
    and accept the variant (different) sequence length among the query, key and value.

    This module does not have args for `module init`.

    `forward()`

    Args:
        query (torch.Tensor): shape [query_tokens, num_head, head_size],
            where tokens is total sequence length among batch size.
        key (torch.Tensor):  shape [key_tokens, num_head, head_size],
            where tokens is total sequence length among batch size.
        value (torch.Tensor): shape [value_tokens, num_head, head_size],
            where tokens is total sequence length among batch size.
        out (torch.Tensor): buffer to get the results, the shape is the same as query.
        seqlen_q (torch.Tensor): shape [batch_size + 1], points the
            current query_tokens among total sequence length.
        seqlen_k (torch.Tensor): shape [batch_size + 1], points the
            current key_tokens among total sequence length.
        max_seqlen_q (int): max/total sequence length of query.
        max_seqlen_k (int): max/total sequence length of key.
        pdropout (float): dropout probability; if greater than 0.0,
            dropout is applied, default is 0.0.
        softmax_scale (float): scaling factor applied is prior to softmax.
        is_causal (bool): whether to apply causal attention masking, default is True.

    Examples:
        >>> # module init:
        >>> varlenAttention_module = ipex.llm.modules.VarlenAttention()
        >>> # forward:
        >>> query = torch.randn(32, 16, 256)
        >>> key = torch.randn(32, 16, 256)
        >>> value = torch.randn(32, 16, 256)
        >>> out = torch.emply_like(query)
        >>> seqlen_q = torch.tensor(1)
        >>> seqlen_k = torch.tensor(1)
        >>> max_seqlen_q = 1
        >>> max_seqlen_k  = 1
        >>> pdropout = 0.0
        >>> softmax_scale  = 0.5
        >>> varlenAttention_module(query, key, value, out, seqlen_q, seqlen_k, max_seqlen_q, max_seqlen_k, pdropout, softmax_scale)

    [Direct function call] This module also provides a `.apply_function`
    function call to apply VarlenAttention without initializing the module.

    The parameters of `apply_function()` are the same as the `forward()` call.

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(self):
        super().__init__()

    @classmethod
    def apply_function(
        cls,
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
        return cls.runtime_ops.get_module_from_device(
            query.device.type, IPEXCustomOpType.VARLEN_ATTENTION, False
        ).apply_function(
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

    def forward(
        self,
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
        runtime_module = self.runtime_ops.get_module_from_device(
            query.device.type, IPEXCustomOpType.VARLEN_ATTENTION, True
        )
        return runtime_module(
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


class PagedAttention:
    r"""
    This module follows the API of two class methods as [vLLM](https://blog.vllm.ai/2023/06/20/vllm.html)
    to enable the paged attention kernel in and use the layout of (num_blocks, num_heads, block_size,  head_size)
    for key/value cache. The basic logic as following figure. Firstly, The DRAM buffer which includes num_blocks
    are pre-allocated to store key or value cache. For every block, block_size tokens can be stored. In the forward
    pass, the cache manager will firstly allocate some slots from this buffer and use reshape_and_cache API to store
    the key/value and then use single_query_cached_kv_attention API to do the scale-dot-product of MHA.
    The block is basic allocation unit of paged attention and the token intra-block are stored one-by-one.
    The block tables are used to map the logical block of sequence into the physical block.

    [class method]: reshape_and_cache
    ipex.llm.modules.PagedAttention.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    This operator is used to store the key/value token states into the pre-allcated kv_cache buffers of paged attention.

    Args:
        key (torch.Tensor): The keytensor. The shape should be [num_seqs, num_heads, head_size].
        value (torch.Tensor): The value tensor. The shape should be [num_seqs, num_heads, head_size].
        key_cache (torch.Tensor):  The pre-allocated buffer to store the key cache.
            The shape should be [num_blocks, block_size, num_heads, head_size].
        value_cache (torch.Tensor): The pre-allocated buffer to store the value cache.
            The shape should be [num_blocks, block_size, num_heads, head_size].
        slot_mapping (torch.Tensor):  It stores the position to store the key/value in the pre-allocated buffers.
            The shape should be the number of sequences. For sequence ``i``, the ``slot_mapping[i] // block_number``
            can get the block index, and the ``slot_mapping % block_size`` can get the offset of this block.

    [class method]: single_query_cached_kv_attention

    .. highlight:: python
    .. code-block:: python

        ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
                                                            out,
                                                            query,
                                                            key_cache,
                                                            value_cache,
                                                            head_mapping,
                                                            scale,
                                                            block_tables,
                                                            context_lens,
                                                            block_size,
                                                            max_context_len,
                                                            alibi_slopes
                                                            )

    This operator is used to be calculated the scale-dot-product based on the paged attention.

    Args:
        out (torch.Tensor): The output tensor with shape of [num_seqs, num_heads, head_size],
            where the num_seqs is the number of the sequence in this batch. The num_heads
            means the number of query head. head_size means the head dimension.
        query (torch.Tensor): The query tensor. The shape should be [num_seqs, num_heads, head_size].
        key_cache (torch.Tensor): The pre-allocated buffer to store the key cache.
            The shape should be [num_blocks,  num_heads, block_size, head_size].
        value_cache(torch.Tensor): The pre-allocated buffer to store the value cache.
            The shape should be [num_blocks, num_heads, block_size, head_size].
        head_mapping(torch.Tensor): The mapping from the query head to the kv head.
            The shape should be the number of query heads.
        scale (float): The scale used by the scale-dot-product.
            In general, it is: ``float(1.0 / (head_size ** 0.5))``.
        block_tables:(torch.Tensor): The mapping table used to mapping the logical sequence
            to the physical sequence. The shape should be [num_seqs, max_num_blocks_per_seq].
        context_lens (torch.Tensor):  The sequence length for every sequence. The size is [num_seqs].
        block_size (int): The block size which means the number of token in every block.
        max_context_len (int): The max sequence length.
        alibi_slopes (torch.Tensor, optinal): which is the alibi slope with the shape of (num_heads).

    [class method]: flash_atten_varlen

    .. highlight:: python
    .. code-block:: python

    ipex.llm.modules.PagedAttention.flash_atten_varlen(
        out,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        scale,
        is_cusal,
        block_tables,
        alibi_slopes
    )

    Args:
        out (torch.Tensor): The output tensor with shape of [num_seqs, num_heads, head_size],
        query (torch.Tensor): The query tensor. The shape should be [num_seqs, num_heads, head_size].
        key_cache (torch.Tensor): The pre-allocated buffer to store the key cache.
            The shape should be [num_blocks,  num_heads, block_size, head_size].
        value_cache(torch.Tensor): The pre-allocated buffer to store the value cache.
            The shape should be [num_blocks, num_heads, block_size, head_size].
        cu_seqlens_q (torch.Tensor): The accumulated sequence length for query. The size is [batch_size+1].
        cu_seqlens_kv (torch.Tensor): The accumulated sequence length for key/value. The size is [batch_size+1].
        max_seqlen_q (int): The max sequence length for query.
        max_seqlen_kv (int): The max sequence length for key/value.
        scale (float): The scale used by the scale-dot-product.
            In general, it is: ``float(1.0 / (head_size ** 0.5))``.
        is_cusal (bool): Whether to apply causal attention masking. Default is True. False is not supported yet.
        block_tables:(torch.Tensor): The mapping table used to mapping the logical sequence
            to the physical sequence. The shape should be [batch_size, max_num_blocks_per_seq].
        alibi_slopes (torch.Tensor, optinal): which is the alibi slope with the shape of (num_heads).

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    @classmethod
    def reshape_and_cache(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        return cls.runtime_ops.get_module_from_device(
            key.device.type, IPEXCustomOpType.PAGED_ATTENTION, False
        ).reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    @classmethod
    def single_query_cached_kv_attention(
        cls,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        head_mapping: torch.Tensor,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: torch.Tensor,
    ):
        return cls.runtime_ops.get_module_from_device(
            output.device.type, IPEXCustomOpType.PAGED_ATTENTION, False
        ).single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @classmethod
    def flash_attn_varlen_func(
        cls,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        scale,
        is_cusal: bool,
        block_tables: torch.Tensor,
        alibi_slopes: torch.Tensor,
    ):
        return cls.runtime_ops.get_module_from_device(
            output.device.type, IPEXCustomOpType.PAGED_ATTENTION, False
        ).flash_attn_varlen_func(
            output,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            scale,
            is_cusal,
            block_tables,
            alibi_slopes,
        )


class IndirectAccessKVCacheAttention(nn.Module):
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

    The shape of the pre-allocated key(value) buffer is [max_seq, beam*batch, head_num, head_size],
    the hidden state of key/value which is the shape of [beam*batch, head_num, head_size] is stored token by token.
    All beam idx information of every timestamp is also stored in a Tensor with the shape of [max_seq, beam*batch].

    `module init`

    Args:
        text_max_length (int) : the max length of kv cache to be used
            for generation (allocate the pre-cache buffer).

    `forward()`

    Args:
        query (torch.Tensor): Query tensor; shape: (beam*batch, seq_len, head_num, head_dim).
        key (torch.Tensor): Key tensor; shape: (beam*batch, seq_len, head_num, head_dim).
        value (torch.Tensor): Value tensor; shape: (beam*batch, seq_len, head_num, head_dim).
        scale_attn (float):scale used by the attention layer. should be ``sqrt(head_size)``.
        layer_past (tuple(torch.Tensor)): tuple(seq_info, key_cache, value_cache, beam-idx).

            - key_cache: key cache tensor, shape: (max_seq, beam*batch,  head_num, head_dim);

            - value_cache: value cache tensor, shape: (max_seq, beam*batch,  head_num, head_dim);

            - beam-idx:  history beam idx, shape:(max_seq, beam*batch);

            - seq_info: Sequence info tensor, shape:(1, 1, max_seq, max_seq).

        head_mask (torch.Tensor): Head mask tensor which is not supported by kernel yet.
        attention_mask(torch.Tensor): Attention mask information.

    Return:
        attn_output: Weighted value which is the output of scale dot product.
        shape (beam*batch, seq_len, head_num, head_size).

        attn_weights: The output tensor of the first matmul in scale dot product
        which is not supported by kernel now.

        new_layer_past: updated layer_past (seq_info, key_cache, value_cache, beam-idx).

    Notes:
        How to reorder KV cache when using the format of IndirectAccessKVCacheAttention (e.g., on llama model
        see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1318)

    .. highlight:: python
    .. code-block:: python

        def _reorder_cache(
            self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        ) -> Tuple[Tuple[torch.Tensor]]:
            if (
                len(past_key_values[0]) == 4 and past_key_values[0][0].shape[-1] == 1
            ):
                for layer_past in past_key_values:
                    layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
                return past_key_values

    [Direct function call] This module also provides a `.apply_function` function call
    to apply IndirectAccessKVCacheAttention without initializing the module.

    The parameters of `apply_function()` are the same as the `forward()` call.

    """

    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(self, text_max_length=2048):
        super().__init__()
        self.text_max_length = text_max_length

    @classmethod
    def apply_function(
        cls,
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
        return cls.runtime_ops.get_module_from_device(
            query.device.type, IPEXCustomOpType.INDIRECTACCESS_KVCACHE_ATTENTION, False
        ).apply_function(
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

    def forward(
        self,
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
    ):
        # query: Tensor [batch, seq_len, num_head, head_dim]
        # key: Tensor [batch, seq_len, num_kv_head, head_dim]
        # value: Tensor [batch, seq_len, num_kv_head, head_dim]
        # layer_past: past key values structure
        # seq_info: Sequence info tensor. The first element should be offset.
        # scale_attn: scale used by the attention layer. should be the sqrt(head_size)
        # head_mask: Head mask tensor which is not support by kernel yet.
        # attention_mask: Attention mask information.

        # Return:
        # attn_output: [batch, seq_len, num_head, head_size]
        # attn_weights: The output tensor of first matmul in scale dot product which is not support by kernel now.
        # key_cache: Key cache tensor [max_seq, batch, seq_len, num_head, head_dim]
        # value_cache: Value cache tensor [max_seq, batch, seq_len, num_head, head_dim]
        # beam-idx: History beam idx [max_seq, batch]

        runtime_module = self.runtime_ops.get_module_from_device(
            query.device.type,
            IPEXCustomOpType.INDIRECTACCESS_KVCACHE_ATTENTION,
            True,
            self.text_max_length,
        )
        return runtime_module(
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
        )
