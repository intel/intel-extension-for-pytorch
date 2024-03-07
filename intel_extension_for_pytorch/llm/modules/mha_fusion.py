import torch
import torch.nn as nn
from typing import Optional, Tuple
from enum import Enum
from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import (
    _IPEXRopeCPU,
    _IPEXRMSNormCPU,
    _IPEXPagedAttentionCPU,
    _IPEXVarlenScaledDotProductCPU,
    _IPEXFastLayerNormCPU,
)

from intel_extension_for_pytorch.transformers.models.xpu.fusions.mha_fusion import (
    _IPEXFastLayerNormXPU,
    _IPEXRopeXPU,
    _IPEXRMSNormXPU,
    _IPEXPagedAttentionXPU,
    _IPEXVarlenScaledDotProductXPU,
)


class IPEXCustomOpType(Enum):
    ROPE: int = 0
    RMS_NORM: int = 1
    PAGED_ATTENTION: int = 2
    FAST_LAYERNORM: int = 3
    VARLEN_ATTENTION: int = 4


CPU_mha_fusion_modules = {
    IPEXCustomOpType.ROPE: _IPEXRopeCPU,
    IPEXCustomOpType.RMS_NORM: _IPEXRMSNormCPU,
    IPEXCustomOpType.PAGED_ATTENTION: _IPEXPagedAttentionCPU,
    IPEXCustomOpType.FAST_LAYERNORM: _IPEXFastLayerNormCPU,
    IPEXCustomOpType.VARLEN_ATTENTION: _IPEXVarlenScaledDotProductCPU,
}

XPU_mha_fusion_modules = {
    IPEXCustomOpType.ROPE: _IPEXRopeXPU,
    IPEXCustomOpType.RMS_NORM: _IPEXRMSNormXPU,
    IPEXCustomOpType.PAGED_ATTENTION: _IPEXPagedAttentionXPU,
    IPEXCustomOpType.FAST_LAYERNORM: _IPEXFastLayerNormXPU,
    IPEXCustomOpType.VARLEN_ATTENTION: _IPEXVarlenScaledDotProductXPU,
}


class IPEXRuntimeCustomOps:
    def __init__(self):
        super().__init__()
        self.device_type = None
        self.runtime_module = None
        self.mha_fusion_modules = {
            "cpu": CPU_mha_fusion_modules,
            "xpu": XPU_mha_fusion_modules,
        }

    def get_module_from_device(
        self,
        device_type: str,
        ops: IPEXCustomOpType,
        is_instance: bool,
        *args,
        **kwargs,
    ):
        if device_type is not self.device_type:
            assert device_type in [
                "cpu",
                "xpu",
            ], f"""The input parameter's device is not supported in ipex, we only support XPU and CPU device,
                "but what we get is {device_type}."""
            if not is_instance:
                self.runtime_module = self.mha_fusion_modules[device_type][ops]
                return self.runtime_module
            else:
                self.runtime_module = self.mha_fusion_modules[device_type][ops](
                    *args, **kwargs
                )
                return self.runtime_module

        return self.runtime_module


class RotaryEmbedding(nn.Module):
    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(
        self,
        max_position_embeddings: int,
        pos_embd_dim: int,
        base=10000,
        backbone: str = None,
    ):
        super().__init__()
        self.model_backbone = backbone
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_dim = pos_embd_dim
        self.base = base

    @classmethod
    def apply(
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
        #    4D: [bs, seqlen, num_head/num_kv_head, head_dim]
        #    3D: [num_tokens, num_head/num_kv_head, head_dim]
        # sin, cos: torch.Tensor [num_tokens, rotary_dim]
        # position_ids (optional): torch.Tensor [bs, seqlen]

        runtime_module = cls.runtime_ops.get_module_from_device(
            query.device.type, IPEXCustomOpType.ROPE, False
        )
        query, key = runtime_module.rotary_embedding(
            query, key, sin, cos, rotary_dim, rotary_half, position_ids
        )
        return query, key

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
        # concat_qkv (in shape) : [bs, seqlen, hidden_size*3]
        # query, key, value (out shape) : [bs, seqlen, num_head/num_kv_head, head_dim]
        # sin, cos: [seqlen, rotary_dim]
        # position_ids: [bs, seqlen]

        # Usage 2 (query, key as input):
        # query/key (in/out shape) : [bs, seqlen, num_head/num_kv_head, head_dim]
        # sin, cos: [seqlen, rotary_dim]
        # position_ids: [bs, seqlen]

        runtime_module = self.runtime_ops.get_module_from_device(
            x.device.type,
            IPEXCustomOpType.ROPE,
            True,
            self.max_position_embeddings,
            self.pos_embd_dim,
            self.base,
            self.model_backbone,
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


class FastLayerNorm(nn.Module):
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
    def apply(cls, hidden_states, normalized_shape, weight, bias, eps):
        return cls.runtime_ops.get_module_from_device(
            hidden_states.device.type, IPEXCustomOpType.FAST_LAYERNORM, False
        ).apply(hidden_states, normalized_shape, weight, bias, eps)

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
    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    def __init__(
        self,
        weight: torch.Tensor = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = weight

    @classmethod
    def apply(cls, hidden_states, weight, eps):
        return cls.runtime_ops.get_module_from_device(
            hidden_states.device.type, IPEXCustomOpType.RMS_NORM, False
        ).apply(hidden_states, weight, eps)

    def forward(self, x: torch.Tensor):
        runtime_module = self.runtime_ops.get_module_from_device(
            x.device.type, IPEXCustomOpType.RMS_NORM, True, self
        )
        return runtime_module(x)


class VarlenAttention(nn.Module):
    runtime_ops: IPEXRuntimeCustomOps = IPEXRuntimeCustomOps()

    @classmethod
    def apply(
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
        ).apply(
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

    def __init__(self):
        super().__init__()

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
