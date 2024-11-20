import torch
import intel_extension_for_pytorch as ipex  # noqa F401
from .RoPE import apply_rotary_pos_emb


def load_attn_fused_qkv_params(self, qkv_layer, out_layer, dtype, grouped=False):
    if dtype == "fp16":
        self.qkv_proj.weight = qkv_layer.weight
        if qkv_layer.bias is not None:
            self.qkv_proj.bias = qkv_layer.bias
            if grouped:
                num_group = self.num_heads // self.num_kv_heads
                self.qkv_proj.bias.data = self.qkv_proj.bias.reshape(
                    self.num_kv_heads, num_group + 2, self.head_dim
                ).contiguous()
            else:
                self.qkv_proj.bias.data = self.qkv_proj.bias.reshape(3, -1).contiguous()

        self.out_proj.weight = out_layer.weight
        self.out_proj.bias = out_layer.bias
    elif dtype == "int4":
        self.qkv_proj_quant.set_weights_bias(qkv_layer.qweight, qkv_layer.bias)
        self.qkv_proj_quant.set_scales_zps_gidx(qkv_layer.scales, qkv_layer.qzeros)
        self.qkv_proj_quant.blocksize = qkv_layer.blocksize

        self.out_proj_quant.set_weights_bias(out_layer.qweight, out_layer.bias)
        self.out_proj_quant.set_scales_zps_gidx(out_layer.scales, out_layer.qzeros)
        self.out_proj_quant.blocksize = out_layer.blocksize

        qkv_layer.qweight = None
        qkv_layer.bias = None
        qkv_layer.scales = None
        qkv_layer.qzeros = None

        out_layer.qweight = None
        out_layer.bias = None
        out_layer.scales = None
        out_layer.qzeros = None


def transpose_attn_fused_qkv_params(self, dtype, grouped=False):
    if dtype == "fp16":
        if grouped:
            num_group = self.num_heads // self.num_kv_heads
            self.qkv_proj.weight.data = (
                self.qkv_proj.weight.reshape(
                    self.num_kv_heads, num_group + 2, self.head_dim, self.embed_dim
                )
                .permute(3, 0, 1, 2)
                .contiguous()
            )
        else:
            self.qkv_proj.weight.data = (
                self.qkv_proj.weight.reshape(3, -1, self.embed_dim)
                .permute(0, 2, 1)
                .contiguous()
            )
        self.out_proj.weight.data = self.out_proj.weight.transpose(0, 1).contiguous()
    elif dtype == "int4":
        if xpu_gemm_use_xetla():
            self.qkv_proj_quant.qweight.data = (
                self.qkv_proj_quant.qweight.t().contiguous()
            )
            self.qkv_proj_quant.use_optimum_format = False
            self.qkv_proj_quant.scales.data = (
                self.qkv_proj_quant.scales.t().contiguous()
            )
            if self.qkv_proj_quant.qzeros is not None:
                self.qkv_proj_quant.qzeros.data = (
                    self.qkv_proj_quant.qzeros.t().contiguous()
                )

            self.out_proj_quant.qweight.data = (
                self.out_proj_quant.qweight.t().contiguous()
            )
            self.out_proj_quant.use_optimum_format = False
            self.out_proj_quant.scales.data = (
                self.out_proj_quant.scales.t().contiguous()
            )
            if self.out_proj_quant.qzeros is not None:
                self.out_proj_quant.qzeros.data = (
                    self.out_proj_quant.qzeros.t().contiguous()
                )
        else:
            self.qkv_proj_quant.qweight.data = (
                self.qkv_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
            )
            self.qkv_proj_quant.use_optimum_format = False
            self.qkv_proj_quant.scales.data = self.qkv_proj_quant.scales

            self.qkv_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")

            self.out_proj_quant.qweight.data = (
                self.out_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
            )
            self.out_proj_quant.use_optimum_format = False
            self.out_proj_quant.scales.data = self.out_proj_quant.scales
            self.out_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
    torch.xpu.synchronize()


def qwen_post_qkv(self, query, key, value, position_ids, layer_past, **kwargs):
    bs_beam, seq, _ = self.get_runtime_shape(query)
    seq = seq if layer_past is None else layer_past[0].size(2) + 1
    rotary_pos_emb_list = kwargs.pop("rotary_pos_emb_list", None)
    if rotary_pos_emb_list is not None:
        if self.is_first_token_beam_search():
            query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb_list)
            self.runtime_cache.key_prompt = key
        else:
            # FIX: need to optimize
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb_list)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            if self.is_beam_search():
                self.runtime_cache.key_cache[
                    self.seq_len - 1 : self.seq_len, :, :, :
                ] = key
            else:
                self.runtime_cache.key_cache[
                    self.prev_seq_len : self.seq_len, :, :, :
                ] = key
    else:
        key_out = self.runtime_cache.key_cache[
            self.prev_seq_len : self.seq_len, :, :, :
        ].view(query.shape)
        self.position_embed(
            query, key, position_ids, self.layer_id, self.beam_size, seq, query, key_out
        )
        key = key_out
    query, key, value = self.combine_kv_cache_interface(query, key, value)
    return query, key, value


def qwen_sdp(self, query, key, value, attention_mask, head_mask, alibi):
    # QWen needs to initilize attention_mask inside attn module
    causal_mask = None
    key_size = key.shape[2]
    if query.shape[2] == key_size:
        causal_mask = torch.tril(
            torch.ones((key_size, key_size), dtype=torch.bool, device=query.device)
        ).view(1, 1, key_size, key_size)
    if attention_mask is not None:
        attention_mask = attention_mask.expand(-1, -1, query.size(2), -1)
        if causal_mask is not None:
            attention_mask = attention_mask.masked_fill(
                ~causal_mask, torch.finfo(query.dtype).min
            )
    else:
        if causal_mask is not None:
            attention_mask = causal_mask
            new_attention_mask = torch.zeros_like(
                attention_mask, dtype=query.dtype, device=query.device
            )
            attention_mask = new_attention_mask.masked_fill_(
                attention_mask.logical_not(), torch.finfo(query.dtype).min
            )
    # Currently only PVC and MTL (without beam search) have sdp fusion available
    if not xpu_sdpa_support(self.is_beam_search(), self.head_dim):
        return self.naive_sdp(query, key, value, attention_mask, head_mask, alibi)
    key, value, key_prompt, value_prompt = self.sdp_kv_preprocess(key, value)
    (
        dropout,
        alpha,
        beta,
        is_causal,
        blocked_attn_mask,
        blocked_alibi,
    ) = self.prepare_sdp_input(query, key, value, attention_mask, alibi)
    attention_output, attn_weight = self.compute_sdp(
        query,
        key,
        value,
        key_prompt,
        value_prompt,
        blocked_attn_mask,
        blocked_alibi,
        head_mask,
        alpha,
        beta,
        dropout,
        is_causal,
    )
    attention_output = self.process_sdp_output(attention_output)
    attention_output = attention_output.reshape(
        attention_output.size()[:-2] + (self.head_dim * self.num_attn_head,)
    )
    return attention_output, attn_weight


def chatglm_post_qkv(self, query, key, value, rotary_pos_emb, layer_past, **kwargs):
    bs_beam, seq, _ = self.get_runtime_shape(query)
    seq = seq if layer_past is None else layer_past[0].size(2) + 1
    query, key = self.position_embed(query, key, rotary_pos_emb)
    query, key, value = self.combine_kv_cache_interface(query, key, value)
    return query, key, value


def chatglm_load_attn_params_grouped(self, qkv_proj, out_proj, dtype):
    if dtype == "fp16":
        embed_dim = self.embed_dim
        num_head = self.num_attn_head
        num_kv_head = self.num_kv_head
        head_dim = self.head_dim
        group = num_head // num_kv_head + 2
        query_weight_size = head_dim * num_head
        key_weight_size = value_weight_size = head_dim * num_kv_head

        def helper(proj, begin, end, g):
            weight = proj.weight[begin:end, ...].view(
                num_kv_head, g, head_dim, embed_dim
            )
            if proj.bias is not None:
                bias = proj.bias[begin:end, ...].view(num_kv_head, g, head_dim)
            else:
                bias = None
            return weight, bias

        wq, bq = helper(qkv_proj, 0, query_weight_size, group - 2)
        wk, bk = helper(
            qkv_proj, query_weight_size, query_weight_size + key_weight_size, 1
        )
        wv, bv = helper(
            qkv_proj, query_weight_size + key_weight_size, qkv_proj.weight.shape[0], 1
        )

        del qkv_proj.weight
        del qkv_proj.bias

        self.qkv_proj.weight = torch.concat([wq, wk, wv], dim=1).contiguous()
        if bq is not None:
            self.qkv_proj.bias = torch.concat([bq, bk, bv], dim=1).contiguous()
        else:
            self.qkv_proj.bias = None

        self.out_proj.weight = out_proj.weight
        self.out_proj.bias = out_proj.bias

        self.position_embed = self.config.rotary_embedding_class(
            self.config, self.qkv_proj.weight.dtype
        )
    elif dtype == "int4":
        self.qkv_proj_quant.set_weights_bias(qkv_proj.qweight, qkv_proj.bias)
        self.qkv_proj_quant.set_scales_zps_gidx(qkv_proj.scales, qkv_proj.qzeros)
        self.qkv_proj_quant.blocksize = qkv_proj.blocksize

        self.out_proj_quant.set_weights_bias(out_proj.qweight, out_proj.bias)
        self.out_proj_quant.set_scales_zps_gidx(out_proj.scales, out_proj.qzeros)
        self.out_proj_quant.blocksize = out_proj.blocksize

        qkv_proj.qweight = None
        qkv_proj.bias = None
        qkv_proj.scales = None
        qkv_proj.qzeros = None

        out_proj.qweight = None
        out_proj.bias = None
        out_proj.scales = None
        out_proj.qzeros = None
        self.position_embed = self.config.rotary_embedding_class(self.config, dtype)


# Use SDPA if XETLA support is available and head_dim is smaller than 128,
# and it's not a beam search when 2D load instruction is not available.
# Use SDPA if XETLA support is available when 2D block array is available.
def xpu_sdpa_support(is_beam_search, head_dim):
    has_2d_block = torch.xpu.has_2d_block_array()
    has_xetla = torch.xpu.has_xetla()

    if not has_2d_block:
        return has_xetla and head_dim <= 128 and not is_beam_search
    else:
        return has_xetla


# There are two implementations of int4 woq gemm.
# Use the onednn implementation on client machines (LNL and BMG) and pvc.
# On other platforms, use the xetla implementation if available
def xpu_gemm_use_xetla():
    has_xmx = torch.xpu.has_xmx()
    has_2d_load = torch.xpu.has_2d_block_array()
    compute_eng = torch.xpu.get_compute_eng()

    if has_xmx and has_2d_load:
        return compute_eng == torch.xpu.XPUComputeEng.XETLA

    return (
        torch.xpu.has_xetla()
        and not torch.xpu.using_onednn_layout()
        and compute_eng
        in (torch.xpu.XPUComputeEng.RECOMMEND, torch.xpu.XPUComputeEng.XETLA)
    )


def need_transpose_contiguous_weight_layout():
    return (
        torch.xpu.has_xetla()
        and (not torch.xpu.using_onednn_layout())
        and (
            torch.xpu.get_compute_eng()
            in (torch.xpu.XPUComputeEng.RECOMMEND, torch.xpu.XPUComputeEng.XETLA)
        )
    )
