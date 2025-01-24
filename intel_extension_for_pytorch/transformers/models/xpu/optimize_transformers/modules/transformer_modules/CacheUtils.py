import torch
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import torch.distributed as dist

enable_ipex_cache_wrapper = False
try:
    from transformers.cache_utils import StaticCache, Cache

    enable_ipex_cache_wrapper = True

    class CacheFormat(Enum):
        # This is the main layout in old ipex optimize transformers flow, this format can make sure token increase
        # at the contiguous dim. this layout would be activate when static cache passed in.
        BFNH = 0
        # This is the layout of old transformers flow, this format are the most popular memory layout in community yet.
        BNFH = 1
        # This is the layout of new transformers cache struct. we are planning to only support this format inorder to
        # fully reuse the cache struct in transformers
        FBNH = 2

    class IPEXStaticCache(Cache):
        def __init__(
            self, cache: StaticCache, cache_format: CacheFormat = CacheFormat.BFNH
        ):
            super().__init__()
            self.max_batch_size = cache.max_batch_size
            self.max_cache_len = cache.max_cache_len
            self.head_dim = cache.head_dim
            self.dtype = cache.dtype
            self.tp_idx = 0
            self.tp_size = 1
            if dist.is_initialized():
                self.tp_size = dist.get_world_size()
                self.tp_idx = dist.get_rank()
            self.num_key_value_heads = cache.num_key_value_heads // self.tp_size
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self.seq_cnt = int(cache.get_seq_length())

            self.key_prompt: List[torch.Tensor] = []
            self.value_prompt: List[torch.Tensor] = []
            self.cache_format = cache_format
            # for now, we only support tensor parallel, so the tp size should equal to world size

            bs = cache.key_cache[0].size(0)
            seqlen = cache.key_cache[0].size(2)

            # adopt tensor parallel on the cache
            for i in range(len(cache.key_cache)):
                start_head = self.tp_idx * self.num_key_value_heads
                end_head = (self.tp_idx + 1) * self.num_key_value_heads
                self.key_cache.append(
                    cache.key_cache[i][:, start_head:end_head, :, :].contiguous()
                )
                self.value_cache.append(
                    cache.value_cache[i][:, start_head:end_head, :, :].contiguous()
                )

            # [b, n, f, h] -> [f, b, n, h]
            if cache_format == CacheFormat.FBNH:
                # keep the memory format as [f, b, n, h]
                for i in range(len(cache.key_cache)):
                    self.key_cache[i] = (
                        self.key_cache[i]
                        .permute(2, 0, 1, 3)
                        .reshape([seqlen, bs, self.num_key_value_heads, self.head_dim])
                        .contiguous()
                    )
                    self.value_cache[i] = (
                        self.value_cache[i]
                        .permute(2, 0, 1, 3)
                        .reshape([seqlen, bs, self.num_key_value_heads, self.head_dim])
                        .contiguous()
                    )
            elif cache_format == CacheFormat.BFNH:
                # keep the memory format as [b, f, n, h]
                for i in range(len(cache.key_cache)):
                    self.key_cache[i] = self.key_cache[i].transpose(1, 2).contiguous()
                    self.value_cache[i] = (
                        self.value_cache[i].transpose(1, 2).contiguous()
                    )
            elif cache_format == CacheFormat.BNFH:
                for i in range(len(cache.key_cache)):
                    self.key_cache[i] = self.key_cache[i]
                    self.value_cache[i] = self.value_cache[i]
            else:
                raise ValueError(f"Unsupported cache format: {cache_format}")

        def update(
            self,
            key_states: List[torch.Tensor],
            value_states: List[torch.Tensor],
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            cache_position = cache_kwargs.get("cache_position")
            beam_search = cache_kwargs.get("beam_search", False)
            num_heads = cache_kwargs.get("num_heads", self.num_key_value_heads)
            # update kv cache size
            # store kv prompt cache if beam search
            seqlen = self.update_or_get_seq_cnt(
                layer_idx, key_states
            ) + key_states.size(2)
            if beam_search and key_states.size(2) > 1:
                self.key_prompt.append(key_states)
                self.value_prompt.append(value_states)
                return key_states, value_states
            # if not prompt in beam search or greedy search, update the seqlen in kv cache
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(
                device=key_states.device
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(
                device=value_states.device
            )
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]

            if cache_position is None:
                k_out.copy_(key_states)
                v_out.copy_(value_states)
            else:
                # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
                # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
                # operation, that avoids copies and uses less memory.
                try:
                    k_out.index_copy_(2, cache_position, key_states)
                    v_out.index_copy_(2, cache_position, value_states)
                except NotImplementedError:
                    # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                    k_out[:, :, cache_position] = key_states
                    v_out[:, :, cache_position] = value_states

            return k_out[:, :, :seqlen, :], v_out[:, :, :seqlen, :]

        def update_or_get_seq_cnt(
            self,
            layer_idx: int = -1,
            key_states: torch.Tensor = None,
        ):
            if key_states is not None and layer_idx == len(self.key_cache) - 1:
                seqlen = key_states.shape[-2]
                self.seq_cnt += seqlen
            return self.seq_cnt

        def get_prompt_for_beam_search(self, layer_idx: int):
            return self.key_prompt[layer_idx], self.value_prompt[layer_idx]

        def get_kv_slice_for_qkv(
            self,
            layer_idx: int,
            cache_position: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            prev_len = self.update_or_get_seq_cnt(layer_idx)
            curr_len = cache_position.size(-1)
            key = self.key_cache[layer_idx][
                prev_len : prev_len + curr_len, :, :, :
            ].view(curr_len, -1, self.num_key_value_heads * self.head_dim)
            value = self.value_cache[layer_idx][
                prev_len : prev_len + curr_len, :, :
            ].view(curr_len, -1, self.num_key_value_heads * self.head_dim)
            return (key, value)

        def get_kv_slice_for_decoding(
            self,
            layer_idx: int,
            key: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # return the key and value cache for decoding
            prompt_len = (
                0 if len(self.key_prompt) == 0 else self.key_prompt[layer_idx].size(2)
            )
            seqlen = self.update_or_get_seq_cnt(layer_idx) + key.size(2)
            if self.cache_format == CacheFormat.FBNH:
                key = self.key_cache[layer_idx][prompt_len:seqlen, :, :, :].permute(
                    1, 2, 0, 3
                )
                value = self.value_cache[layer_idx][prompt_len:seqlen, :, :, :].permute(
                    1, 2, 0, 3
                )
            elif self.cache_format == CacheFormat.BNFH:
                key = self.key_cache[layer_idx][:, :, prompt_len:seqlen, :]
                value = self.value_cache[layer_idx][:, :, prompt_len:seqlen, :]
            elif self.cache_format == CacheFormat.BFNH:
                key = self.key_cache[layer_idx][:, prompt_len:seqlen, :, :].permute(
                    0, 2, 1, 3
                )
                value = self.value_cache[layer_idx][:, prompt_len:seqlen, :, :].permute(
                    0, 2, 1, 3
                )
            return key, value

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            """Returns the sequence length of the cached states that were seen by the model."""
            # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
            # limit the check to the first batch member and head dimension.
            # TODO: deprecate this function in favor of `cache_position`
            return self.seq_cnt

        def get_prompt_len(self, layer_idx: Optional[int] = 0) -> int:
            """return the sequence length of the prompt for beam search"""
            return (
                self.key_prompt[layer_idx].size(2)
                if len(self.key_prompt) > layer_idx
                else 0
            )

        def get_max_length(self) -> Optional[int]:
            """Returns the maximum sequence length of the cached states."""
            return self.max_cache_len

        def reset(self):
            """Resets the cache values while preserving the objects"""
            for layer_idx in range(len(self.key_cache)):
                # In-place ops prevent breaking the static address
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()

except ImportError:
    pass


def warp_cache_if_needed(cache):
    if enable_ipex_cache_wrapper and isinstance(cache, StaticCache):
        return IPEXStaticCache(cache, CacheFormat.FBNH)
    return cache
