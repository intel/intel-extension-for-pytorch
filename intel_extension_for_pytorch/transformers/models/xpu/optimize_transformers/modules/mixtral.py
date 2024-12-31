import torch
import intel_extension_for_pytorch as ipex  # noqa F401
from typing import Optional, Tuple
from torch import nn
import torch.distributed as dist

from .transformer_modules.RoPE import MixtralRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm

from .transformer_modules.Linear import (  # noqa F401
    IPEXTransformerLinear,
)  # noqa

from ._transformers import MAX_OUT_SEQ_LEN

from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation

from .transformer_modules.DecoderBlock import IPEXTransformerBlock

from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.Activation import ACT2FN

import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


def ref_moe_gemm(matrix_a, matrix_b, rows_for_experts, rows_for_experts_cpu, n_experts):
    total_m = matrix_a.shape[0]
    gemm_k = matrix_a.shape[1]
    gemm_n = matrix_b.shape[2]

    output = torch.zeros(total_m, gemm_n, device=matrix_a.device, dtype=matrix_a.dtype)
    start = 0
    for i in range(n_experts):
        end = start + rows_for_experts_cpu[i].item()
        output[start:end] = torch.mm(matrix_a[start:end], matrix_b[i])
        start = end

    return output


def ref_topk_softmax(gating_logits, n_topk):
    input_dtype = gating_logits.dtype
    gating_logits = gating_logits.to(torch.float)
    softmax = torch.nn.functional.softmax(gating_logits, dim=-1, dtype=torch.float)
    topk_weights, topk_indices = torch.topk(softmax, n_topk, dim=-1)

    # token_for_expert: # of tokens for each expert
    # token_offset: the offset of each token for each export
    n_experts = gating_logits.shape[-1]
    token_for_experts = torch.zeros(
        n_experts, device=gating_logits.device, dtype=torch.int32
    )

    token_offset = torch.empty(
        topk_weights.shape, device=gating_logits.device, dtype=torch.int32
    )
    n_tokens = gating_logits.shape[0]
    for i in range(n_tokens):
        for j in range(n_topk):
            expert_id = topk_indices[i, j]
            token_for_experts[expert_id] += 1
            token_offset[i, j] = token_for_experts[expert_id] - 1

    return (
        topk_weights.to(torch.float),
        topk_indices.to(torch.int32),
        token_for_experts,
        token_offset,
    )


def ref_moe_scatter(
    hidden_state, token_for_experts, topk_indices, token_offset, num_experts, n_topk
):
    # inclusive scan
    n_token, n_channels = hidden_state.shape
    reorder_hidden_state = torch.empty(
        (n_token * n_topk, n_channels),
        dtype=hidden_state.dtype,
        device=hidden_state.device,
    )
    mapped_slot = torch.empty(
        (n_token, n_topk), dtype=torch.int32, device=topk_indices.device
    )

    token_for_experts = torch.cumsum(token_for_experts, dim=0, dtype=torch.int32)
    for i in range(n_token):
        for j in range(n_topk):
            expert_id = topk_indices[i, j]
            expert_offset = token_for_experts[expert_id - 1] if expert_id > 0 else 0
            slot_id = token_offset[i, j] + expert_offset
            mapped_slot[i, j] = slot_id
            reorder_hidden_state[slot_id] = hidden_state[i]
    return reorder_hidden_state, mapped_slot


def ref_moe_gather(
    reorder_hidden_state,
    topk_weights,
    mapped_slot,
    token_for_experts,
    n_expert,
    n_topk,
    normalize_scale,
):
    # normalize topk_weights along the last dimension
    n_token = topk_weights.shape[0]
    n_channels = reorder_hidden_state.shape[-1]
    if normalize_scale:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    gathered = torch.zeros(
        (n_token, n_channels),
        dtype=reorder_hidden_state.dtype,
        device=reorder_hidden_state.device,
    )
    for i in range(n_token):
        for j in range(n_topk):
            slot_id = mapped_slot[i, j]
            gathered[i] += topk_weights[i, j] * reorder_hidden_state[slot_id]
    return gathered


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_dim
        self.hidden_dim = config.embedding_dim

        self.moe_w1 = IPEXTransformerLinear()
        self.moe_w2 = IPEXTransformerLinear()
        # self.moe_w3 = IPEXTransformerLinear()
        self.num_experts = config.num_local_experts

        self.act_fn = ACT2FN[config.activation_function]

    def linear_silu_mul(self, x):
        half = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (half,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return torch.ops.torch_ipex.silu_and_mul(x, out)

    def forward(
        self,
        hidden_states,
        rows_for_experts=None,
        rows_for_experts_cpu=None,
        use_optimized=False,
        residual=None,
    ):
        hidden_states = torch.ops.torch_ipex.moe_gemm(
            hidden_states,
            self.moe_w1.weight,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
        )
        hidden_states = self.linear_silu_mul(hidden_states)
        hidden_states = torch.ops.torch_ipex.moe_gemm(
            hidden_states,
            self.moe_w2.weight,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
        )
        return hidden_states


class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40."
        )
        super().__init__(*args, **kwargs)


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.embedding_dim
        self.ffn_dim = config.intermediate_dim
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.tp_group = config.tp_group

        # gating
        # self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, device="xpu")
        self.moe_gate = IPEXTransformerLinear()

        # self.moe_experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        self.fused_moe_experts = MixtralBlockSparseTop2MLP(config)

    def all_reduce_if_necessary(self, target):
        if self.tp_group is not None:
            dist.all_reduce(target, group=self.tp_group)
        return target

    def load_parameter(self, expert_weights_w1, expert_weights_w2, gate_weight):
        self.moe_gate.weight = gate_weight
        self.fused_moe_experts.moe_w1.weight = expert_weights_w1
        self.fused_moe_experts.moe_w2.weight = expert_weights_w2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.moe_gate(hidden_states)

        # --------- fusion:  topk softmax  -------------------
        routing_weights, selected_experts, rows_for_experts, expert_offsets = (
            torch.ops.torch_ipex.topk_softmax(router_logits, self.top_k)
        )

        # --------- fusion: scatter + moegemm + gather -------------------
        # scatter hidden_states such that the token stride for each expert's input is contiguous
        reordered_hidden_states, mapped_slot = torch.ops.torch_ipex.moe_scatter(
            hidden_states,
            rows_for_experts,
            selected_experts,
            expert_offsets,
            self.num_experts,
            self.top_k,
        )

        # moe gemm
        # TODO: remove this memcpy
        rows_for_experts_cpu = rows_for_experts.to("cpu")
        reordered_moe_output = self.fused_moe_experts(
            reordered_hidden_states,
            rows_for_experts,
            rows_for_experts_cpu,
            use_optimized=True,
        )

        # Re-gather the outputs of MoE and scale them by the gating score, note this kernel also reset rows_for_experts to zero
        normalize_scale = True
        moe_output = torch.ops.torch_ipex.moe_gather(
            reordered_moe_output,
            routing_weights,
            mapped_slot,
            rows_for_experts,
            self.num_experts,
            self.top_k,
            normalize_scale,
        )
        self.all_reduce_if_necessary(moe_output)
        moe_output = moe_output.reshape(batch_size, sequence_length, hidden_dim)
        return moe_output, router_logits


class NewIPEXMixtralBlock(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
        **kwargs,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        if dtype == "fp16" or dtype == "bf16":
            self.self_attn = IPEXAttention(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEX Mixtral dose not support this modelType {} !".format(dtype)
            )
        self.block_sparse_moe = MixtralSparseMoeBlock(self.ipex_config)

        self.input_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

    def build_sparse_moe_from_config(self):
        return

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.hidden_act
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation function,"
            "can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfig(
            num_experts_per_tok=self.config.num_experts_per_tok,
            num_local_experts=self.config.num_local_experts,
            output_router_logits=self.config.output_router_logits,
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=self.config.max_position_embeddings,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=MixtralRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=self.config.hidden_act,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            residual_dropout=None,
            attn_dropout=None,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            positional_embedding_base=self.config.rope_theta,
            device=self.device,
            dtype=dtype,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.self_attn.load_parameter(
            self.module.self_attn.q_proj,
            self.module.self_attn.k_proj,
            self.module.self_attn.v_proj,
            self.module.self_attn.o_proj,
        )

    # TODO: we currently move tranpose to port part, as tranpose a 3d tensor will increase memory usage
    # fix this once we know why it's different with 2d
    def port_moe_parameter(self, transpose=True):
        expert_weights_w1 = []
        expert_weights_w2 = []
        for i in range(self.ipex_config.num_local_experts):
            concated_w1 = torch.concat(
                (
                    self.module.block_sparse_moe.experts[i].w1.weight,
                    self.module.block_sparse_moe.experts[i].w3.weight,
                ),
                dim=0,
            )
            w2 = self.module.block_sparse_moe.experts[i].w2.weight
            if transpose:
                concated_w1 = concated_w1.transpose(0, 1).contiguous()
                w2 = w2.transpose(0, 1).contiguous()
            expert_weights_w1.append(concated_w1)
            expert_weights_w2.append(w2)
        expert_weights_w1 = torch.stack(expert_weights_w1, dim=0)
        expert_weights_w2 = torch.stack(expert_weights_w2, dim=0)
        wgate = (
            self.module.block_sparse_moe.gate.weight.transpose(0, 1).contiguous()
            if transpose
            else self.module.block_sparse_moe.gate.weight
        )
        self.block_sparse_moe.load_parameter(
            expert_weights_w1, expert_weights_w2, wgate
        )
        # rewrite original weight to avoid memory pressure
        # expert_weights_w1: [num_experts, in_channels, hidden_size * 2]
        # expert_weights_w2: [num_experts, hidden_size, in_channels]
        hidden_size = expert_weights_w1.shape[-1] // 2
        for i in range(self.ipex_config.num_local_experts):
            self.module.block_sparse_moe.experts[i].w1.weight.data = expert_weights_w1[
                i, :, :hidden_size
            ]
            self.module.block_sparse_moe.experts[i].w3.weight.data = expert_weights_w1[
                i, :, hidden_size:
            ]
            self.module.block_sparse_moe.experts[i].w2.weight.data = expert_weights_w2[
                i, :, :
            ]
        self.module.block_sparse_moe.gate.weight.data = wgate
        torch.xpu.empty_cache()

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def transpose_parameter(self):
        self.self_attn.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        self.port_norm_parameter()
        self.port_attn_parameter()
        self.port_moe_parameter(self.ipex_config.transpose)

        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.self_attn.cat_qkv()
        torch.xpu.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            residual=residual,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
        return outputs
