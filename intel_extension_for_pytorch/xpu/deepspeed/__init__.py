import torch
import intel_extension_for_pytorch as ipex

__all__ = [
    "has_deepspeed",
    "ds_quantize_fp32",
    "ds_quantize_fp16",
    "ds_sr_quantize_fp32",
    "ds_sr_quantize_fp16",
    "ds_quantize_asym_fp32",
    "ds_quantize_asym_fp16",
    "ds_sr_quantize_asym_fp32",
    "ds_sr_quantize_asym_fp16",
    "quantize",
    "dequantize",
    "dequantize_fp32",
    "dequantize_int4_to_half_experimental",
    "dequantize_int8_to_half_experimental",
    "swizzle_quant",
    "quantized_reduction",
    "softmax_context_int8",
    "gated_activation",
    "layer_norm",
    "_layer_norm_residual",
    "layer_norm_residual_store_pre_ln_res",
    "ds_rms_norm",
    "ds_pre_rms_norm",
    "_vector_add",
    "apply_rotary_pos_emb",
    "moe_res_matmul",
    "reset_cache",
    "release_workspace",
    "retake_workspace",
    "softmax_fp32",
    "softmax_fp16",
    "softmax_bf16",
    "softmax_context_fp32",
    "softmax_context_fp16",
    "softmax_context_bf16",
    "bias_add_fp32",
    "bias_add_fp16",
    "bias_add_bf16",
    "bias_relu_fp32",
    "bias_relu_fp16",
    "bias_relu_bf16",
    "bias_residual_fp32",
    "bias_residual_fp16",
    "bias_residual_bf16",
    "qkv_gemm_fp32",
    "qkv_gemm_fp16",
    "qkv_gemm_bf16",
    "rms_qkv_gemm_fp32",
    "rms_qkv_gemm_fp16",
    "rms_qkv_gemm_bf16",
    "mlp_gemm_fp32",
    "mlp_gemm_fp16",
    "mlp_gemm_bf16",
    "rms_mlp_gemm_fp32",
    "rms_mlp_gemm_fp16",
    "rms_mlp_gemm_bf16",
    "vector_matmul_fp32",
    "vector_matmul_fp16",
    "vector_matmul_bf16",
    "linear_layer_fp32",
    "linear_layer_fp16",
    "linear_layer_bf16",
    "fused_gemm_gelu_fp32",
    "fused_gemm_gelu_fp16",
    "fused_gemm_gelu_bf16",
    "residual_add_bias_fp32",
    "residual_add_bias_fp16",
    "residual_add_bias_bf16",
    "einsum_sec_sm_ecm_fp32",
    "einsum_sec_sm_ecm_fp16",
    "einsum_sec_sm_ecm_bf16",
    "add_padding_fp32",
    "add_padding_fp16",
    "add_padding_bf16",
    "pad_transform_fp32",
    "pad_transform_fp16",
    "pad_transform_bf16",
    "allocate_workspace_fp32",
    "allocate_workspace_fp16",
    "allocate_workspace_bf16",
    "ti_dequantize_fp32",
    "ti_dequantize_fp16",
    "ti_dequantize_bf16",
]


def has_deepspeed():
    return ipex._C._is_ds_kernel_enabled()


def ds_quantize_fp32(input, groups, bit):
    return torch.ops.torch_ipex.ds_quantize_fp32(input, groups, bit)


def ds_quantize_fp16(input, groups, bit):
    return torch.ops.torch_ipex.ds_quantize_fp16(input, groups, bit)


def ds_sr_quantize_fp32(vals, groups, bits):
    return torch.ops.torch_ipex.ds_sr_quantize_fp32(vals, groups, bits)


def ds_sr_quantize_fp16(vals, groups, bits):
    return torch.ops.torch_ipex.ds_sr_quantize_fp16(vals, groups, bits)


def ds_quantize_asym_fp32(vals, groups, bits):
    return torch.ops.torch_ipex.ds_quantize_asym_fp32(vals, groups, bits)


def ds_quantize_asym_fp16(vals, groups, bits):
    return torch.ops.torch_ipex.ds_quantize_asym_fp16(vals, groups, bits)


def ds_sr_quantize_asym_fp32(vals, groups, bits):
    return torch.ops.torch_ipex.ds_sr_quantize_asym_fp32(vals, groups, bits)


def ds_sr_quantize_asym_fp16(vals, groups, bits):
    return torch.ops.torch_ipex.ds_sr_quantize_asym_fp16(vals, groups, bits)


def quantize(input_vals, groups, numBits, is_symmetric):
    return torch.ops.torch_ipex.quantize(input_vals, groups, numBits, is_symmetric)


def dequantize(activations, params, num_groups, q_bits, is_symmetric):
    return torch.ops.torch_ipex.dequantize(
        activations, params, num_groups, q_bits, is_symmetric
    )


def dequantize_fp32(activations, params, num_groups, q_bits, is_symmetric):
    return torch.ops.torch_ipex.dequantize_fp32(
        activations, params, num_groups, q_bits, is_symmetric
    )


def dequantize_int4_to_half_experimental(
    data_in, scale_buffer, min_val_buffer, num_group, group_size
):
    return torch.ops.torch_ipex.dequantize_int4_to_half_experimental(
        data_in, scale_buffer, min_val_buffer, num_group, group_size
    )


def dequantize_int8_to_half_experimental(
    data_in, scale_buffer, min_val_buffer, num_group, group_size
):
    return torch.ops.torch_ipex.dequantize_int8_to_half_experimental(
        data_in, scale_buffer, min_val_buffer, num_group, group_size
    )


def swizzle_quant(
    input_vals, groups, num_bits, isSymmetric, pipeline_size, nodes, devices_per_node
):
    return torch.ops.torch_ipex.swizzle_quant(
        input_vals,
        groups,
        num_bits,
        isSymmetric,
        pipeline_size,
        nodes,
        devices_per_node,
    )


def quantized_reduction(
    input_vals,
    input_scales,
    in_groups,
    out_groups,
    num_bits,
    isSymmetric,
    devices_per_node,
):
    return torch.ops.torch_ipex.quantized_reduction(
        input_vals,
        input_scales,
        in_groups,
        out_groups,
        num_bits,
        isSymmetric,
        devices_per_node,
    )


def softmax_context_int8(
    query,
    prev_key,
    new_key,
    attn_mask,
    prev_value,
    new_value,
    heads,
    norm_factor,
    merging,
    triangular,
    local_attention,
    window_size,
    no_masking,
):
    return torch.ops.torch_ipex.softmax_context_int8(
        query,
        prev_key,
        new_key,
        attn_mask,
        prev_value,
        new_value,
        heads,
        norm_factor,
        merging,
        triangular,
        local_attention,
        window_size,
        no_masking,
    )


def gated_activation(input, gate, mode):
    return torch.ops.torch_ipex.gated_activation(input, gate, mode)


def layer_norm(input, gamma, beta, epsilon):
    return torch.ops.torch_ipex.layer_norm(input, gamma, beta, epsilon)


def _layer_norm_residual(input, bias, residual, gamma, beta, epsilon):
    return torch.ops.torch_ipex._layer_norm_residual(
        input, bias, residual, gamma, beta, epsilon
    )


def layer_norm_residual_store_pre_ln_res(input, bias, residual, gamma, beta, epsilon):
    return torch.ops.torch_ipex.layer_norm_residual_store_pre_ln_res(
        input, bias, residual, gamma, beta, epsilon
    )


def ds_rms_norm(input, gamma, epsilon):
    return torch.ops.torch_ipex.ds_rms_norm(input, gamma, epsilon)


def ds_pre_rms_norm(input, residual, gamma, epsilon):
    return torch.ops.torch_ipex.ds_pre_rms_norm(input, residual, gamma, epsilon)


def _vector_add(a, b, gamma):
    return torch.ops.torch_ipex._vector_add(a, b, gamma)


def apply_rotary_pos_emb(
    mixed_query, key_layer, rotary_dim, offset, num_heads, rotate_half, rop_theta
):
    return torch.ops.torch_ipex.apply_rotary_pos_emb(
        mixed_query, key_layer, rotary_dim, offset, num_heads, rotate_half, rop_theta
    )


def moe_res_matmul(moe_res, coef, output):
    return torch.ops.torch_ipex.moe_res_matmul(moe_res, coef, output)


def reset_cache():
    torch.ops.torch_ipex.reset_cache()


def release_workspace():
    torch.ops.torch_ipex.release_workspace()


def retake_workspace():
    torch.ops.torch_ipex.retake_workspace()


def softmax_fp32(
    attn_scores,
    attn_mask,
    alibi,
    triangular,
    recompute,
    local_attention,
    window_size,
    async_op,
    layer_scale,
    head_offset,
    mp_size,
):
    return torch.ops.torch_ipex.softmax_fp32(
        attn_scores,
        attn_mask,
        alibi,
        triangular,
        recompute,
        local_attention,
        window_size,
        async_op,
        layer_scale,
        head_offset,
        mp_size,
    )


def softmax_fp16(
    attn_scores,
    attn_mask,
    alibi,
    triangular,
    recompute,
    local_attention,
    window_size,
    async_op,
    layer_scale,
    head_offset,
    mp_size,
):
    return torch.ops.torch_ipex.softmax_fp16(
        attn_scores,
        attn_mask,
        alibi,
        triangular,
        recompute,
        local_attention,
        window_size,
        async_op,
        layer_scale,
        head_offset,
        mp_size,
    )


def softmax_bf16(
    attn_scores,
    attn_mask,
    alibi,
    triangular,
    recompute,
    local_attention,
    window_size,
    async_op,
    layer_scale,
    head_offset,
    mp_size,
):
    return torch.ops.torch_ipex.softmax_bf16(
        attn_scores,
        attn_mask,
        alibi,
        triangular,
        recompute,
        local_attention,
        window_size,
        async_op,
        layer_scale,
        head_offset,
        mp_size,
    )


def softmax_context_fp32(
    query_key_value,
    attn_mask,
    rotary_dim,
    rotate_half,
    rotate_every_two,
    heads,
    num_kv,
    norm_factor,
    triangular,
    local_attention,
    window_size,
    no_masking,
    layer_id,
    num_layers,
    alibi,
    rop_theta,
):
    return torch.ops.torch_ipex.softmax_context_fp32(
        query_key_value,
        attn_mask,
        rotary_dim,
        rotate_half,
        rotate_every_two,
        heads,
        num_kv,
        norm_factor,
        triangular,
        local_attention,
        window_size,
        no_masking,
        layer_id,
        num_layers,
        alibi,
        rop_theta,
    )


def softmax_context_fp16(
    query_key_value,
    attn_mask,
    rotary_dim,
    rotate_half,
    rotate_every_two,
    heads,
    num_kv,
    norm_factor,
    triangular,
    local_attention,
    window_size,
    no_masking,
    layer_id,
    num_layers,
    alibi,
    rop_theta,
):
    return torch.ops.torch_ipex.softmax_context_fp16(
        query_key_value,
        attn_mask,
        rotary_dim,
        rotate_half,
        rotate_every_two,
        heads,
        num_kv,
        norm_factor,
        triangular,
        local_attention,
        window_size,
        no_masking,
        layer_id,
        num_layers,
        alibi,
        rop_theta,
    )


def softmax_context_bf16(
    query_key_value,
    attn_mask,
    rotary_dim,
    rotate_half,
    rotate_every_two,
    heads,
    num_kv,
    norm_factor,
    triangular,
    local_attention,
    window_size,
    no_masking,
    layer_id,
    num_layers,
    alibi,
    rop_theta,
):
    return torch.ops.torch_ipex.softmax_context_bf16(
        query_key_value,
        attn_mask,
        rotary_dim,
        rotate_half,
        rotate_every_two,
        heads,
        num_kv,
        norm_factor,
        triangular,
        local_attention,
        window_size,
        no_masking,
        layer_id,
        num_layers,
        alibi,
        rop_theta,
    )


def bias_gelu_fp32(input, bias):
    return torch.ops.torch_ipex.bias_gelu_fp32(input, bias)


def bias_gelu_fp16(input, bias):
    return torch.ops.torch_ipex.bias_gelu_fp16(input, bias)


def bias_gelu_bf16(input, bias):
    return torch.ops.torch_ipex.bias_gelu_bf16(input, bias)


def bias_add_fp32(input, bias):
    return torch.ops.torch_ipex.bias_add_fp32(input, bias)


def bias_add_fp16(input, bias):
    return torch.ops.torch_ipex.bias_add_fp16(input, bias)


def bias_add_bf16(input, bias):
    return torch.ops.torch_ipex.bias_add_bf16(input, bias)


def bias_relu_fp32(input, bias):
    return torch.ops.torch_ipex.bias_relu_fp32(input, bias)


def bias_relu_fp16(input, bias):
    return torch.ops.torch_ipex.bias_relu_fp16(input, bias)


def bias_relu_bf16(input, bias):
    return torch.ops.torch_ipex.bias_relu_bf16(input, bias)


def bias_residual_fp32(input, residual, bias):
    return torch.ops.torch_ipex.bias_residual_fp32(input, residual, bias)


def bias_residual_fp16(input, residual, bias):
    return torch.ops.torch_ipex.bias_residual_fp16(input, residual, bias)


def bias_residual_bf16(input, residual, bias):
    return torch.ops.torch_ipex.bias_residual_bf16(input, residual, bias)


def qkv_gemm_fp32(
    input,
    weight,
    q_scale,
    bias,
    gamma,
    beta,
    epsilon,
    add_bias,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.qkv_gemm_fp32(
        input,
        weight,
        q_scale,
        bias,
        gamma,
        beta,
        epsilon,
        add_bias,
        q_int8,
        transposed_mode,
    )


def qkv_gemm_fp16(
    input,
    weight,
    q_scale,
    bias,
    gamma,
    beta,
    epsilon,
    add_bias,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.qkv_gemm_fp16(
        input,
        weight,
        q_scale,
        bias,
        gamma,
        beta,
        epsilon,
        add_bias,
        q_int8,
        transposed_mode,
    )


def qkv_gemm_bf16(
    input,
    weight,
    q_scale,
    bias,
    gamma,
    beta,
    epsilon,
    add_bias,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.qkv_gemm_bf16(
        input,
        weight,
        q_scale,
        bias,
        gamma,
        beta,
        epsilon,
        add_bias,
        q_int8,
        transposed_mode,
    )


def rms_qkv_gemm_fp32(input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode):
    return torch.ops.torch_ipex.rms_qkv_gemm_fp32(
        input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode
    )


def rms_qkv_gemm_fp16(input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode):
    return torch.ops.torch_ipex.rms_qkv_gemm_fp16(
        input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode
    )


def rms_qkv_gemm_bf16(input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode):
    return torch.ops.torch_ipex.rms_qkv_gemm_bf16(
        input, weight, q_scale, gamma, epsilon, q_int8, transposed_mode
    )


def mlp_gemm_fp32(
    input,
    residual,
    input_bias,
    weight_interm,
    weight_out,
    bias,
    gamma,
    beta,
    epsilon,
    preLayerNorm,
    mlp_after_attn,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.mlp_gemm_fp32(
        input,
        residual,
        input_bias,
        weight_interm,
        weight_out,
        bias,
        gamma,
        beta,
        epsilon,
        preLayerNorm,
        mlp_after_attn,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def mlp_gemm_fp16(
    input,
    residual,
    input_bias,
    weight_interm,
    weight_out,
    bias,
    gamma,
    beta,
    epsilon,
    preLayerNorm,
    mlp_after_attn,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.mlp_gemm_fp16(
        input,
        residual,
        input_bias,
        weight_interm,
        weight_out,
        bias,
        gamma,
        beta,
        epsilon,
        preLayerNorm,
        mlp_after_attn,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def mlp_gemm_bf16(
    input,
    residual,
    input_bias,
    weight_interm,
    weight_out,
    bias,
    gamma,
    beta,
    epsilon,
    preLayerNorm,
    mlp_after_attn,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.mlp_gemm_bf16(
        input,
        residual,
        input_bias,
        weight_interm,
        weight_out,
        bias,
        gamma,
        beta,
        epsilon,
        preLayerNorm,
        mlp_after_attn,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def rms_mlp_gemm_fp32(
    input,
    residual,
    weight_interm,
    weight_out,
    gamma,
    epsilon,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.rms_mlp_gemm_fp32(
        input,
        residual,
        weight_interm,
        weight_out,
        gamma,
        epsilon,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def rms_mlp_gemm_fp16(
    input,
    residual,
    weight_interm,
    weight_out,
    gamma,
    epsilon,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.rms_mlp_gemm_fp16(
        input,
        residual,
        weight_interm,
        weight_out,
        gamma,
        epsilon,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def rms_mlp_gemm_bf16(
    input,
    residual,
    weight_interm,
    weight_out,
    gamma,
    epsilon,
    q_scale,
    q_scale1,
    q_int8,
    activation_type,
    transposed_mode,
):
    return torch.ops.torch_ipex.rms_mlp_gemm_bf16(
        input,
        residual,
        weight_interm,
        weight_out,
        gamma,
        epsilon,
        q_scale,
        q_scale1,
        q_int8,
        activation_type,
        transposed_mode,
    )


def vector_matmul_fp32(input, weight, async_op, q_scale, q_int8, transposed_mode):
    return torch.ops.torch_ipex.vector_matmul_fp32(
        input, weight, async_op, q_scale, q_int8, transposed_mode
    )


def vector_matmul_fp16(input, weight, async_op, q_scale, q_int8, transposed_mode):
    return torch.ops.torch_ipex.vector_matmul_fp16(
        input, weight, async_op, q_scale, q_int8, transposed_mode
    )


def vector_matmul_bf16(input, weight, async_op, q_scale, q_int8, transposed_mode):
    return torch.ops.torch_ipex.vector_matmul_bf16(
        input, weight, async_op, q_scale, q_int8, transposed_mode
    )


def linear_layer_fp32(
    input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta
):
    return torch.ops.torch_ipex.linear_layer_fp32(
        input,
        weight,
        bias,
        add_bias,
        do_flash_attn,
        num_heads,
        transposed_mode,
        rope_theta,
    )


def linear_layer_fp16(
    input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta
):
    return torch.ops.torch_ipex.linear_layer_fp16(
        input,
        weight,
        bias,
        add_bias,
        do_flash_attn,
        num_heads,
        transposed_mode,
        rope_theta,
    )


def linear_layer_bf16(
    input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta
):
    return torch.ops.torch_ipex.linear_layer_bf16(
        input,
        weight,
        bias,
        add_bias,
        do_flash_attn,
        num_heads,
        transposed_mode,
        rope_theta,
    )


def fused_gemm_gelu_fp32(
    input,
    weight,
    weight_scale,
    bias,
    weight_out,
    weight_out_scale,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.fused_gemm_gelu_fp32(
        input,
        weight,
        weight_scale,
        bias,
        weight_out,
        weight_out_scale,
        q_int8,
        transposed_mode,
    )


def fused_gemm_gelu_fp16(
    input,
    weight,
    weight_scale,
    bias,
    weight_out,
    weight_out_scale,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.fused_gemm_gelu_fp16(
        input,
        weight,
        weight_scale,
        bias,
        weight_out,
        weight_out_scale,
        q_int8,
        transposed_mode,
    )


def fused_gemm_gelu_bf16(
    input,
    weight,
    weight_scale,
    bias,
    weight_out,
    weight_out_scale,
    q_int8,
    transposed_mode,
):
    return torch.ops.torch_ipex.fused_gemm_gelu_bf16(
        input,
        weight,
        weight_scale,
        bias,
        weight_out,
        weight_out_scale,
        q_int8,
        transposed_mode,
    )


def residual_add_bias_fp32(
    hidden_state,
    residual,
    attention_output,
    attention_bias,
    final_bias,
    mp_size,
    mlp_after_attn,
    add_bias,
    preln,
):
    return torch.ops.torch_ipex.residual_add_bias_fp32(
        hidden_state,
        residual,
        attention_output,
        attention_bias,
        final_bias,
        mp_size,
        mlp_after_attn,
        add_bias,
        preln,
    )


def residual_add_bias_fp16(
    hidden_state,
    residual,
    attention_output,
    attention_bias,
    final_bias,
    mp_size,
    mlp_after_attn,
    add_bias,
    preln,
):
    return torch.ops.torch_ipex.residual_add_bias_fp16(
        hidden_state,
        residual,
        attention_output,
        attention_bias,
        final_bias,
        mp_size,
        mlp_after_attn,
        add_bias,
        preln,
    )


def residual_add_bias_bf16(
    hidden_state,
    residual,
    attention_output,
    attention_bias,
    final_bias,
    mp_size,
    mlp_after_attn,
    add_bias,
    preln,
):
    return torch.ops.torch_ipex.residual_add_bias_bf16(
        hidden_state,
        residual,
        attention_output,
        attention_bias,
        final_bias,
        mp_size,
        mlp_after_attn,
        add_bias,
        preln,
    )


def einsum_sec_sm_ecm_fp32(Q, W):
    return torch.ops.torch_ipex.einsum_sec_sm_ecm_fp32(Q, W)


def einsum_sec_sm_ecm_fp16(Q, W):
    return torch.ops.torch_ipex.einsum_sec_sm_ecm_fp16(Q, W)


def einsum_sec_sm_ecm_bf16(Q, W):
    return torch.ops.torch_ipex.einsum_sec_sm_ecm_bf16(Q, W)


def add_padding_fp32(query, key, value):
    return torch.ops.torch_ipex.add_padding_fp32(query, key, value)


def add_padding_fp16(query, key, value):
    return torch.ops.torch_ipex.add_padding_fp16(query, key, value)


def add_padding_bf16(query, key, value):
    return torch.ops.torch_ipex.add_padding_bf16(query, key, value)


def pad_transform_fp32(query, key, value, heads, add_padding):
    return torch.ops.torch_ipex.pad_transform_fp32(
        query, key, value, heads, add_padding
    )


def pad_transform_fp16(query, key, value, heads, add_padding):
    return torch.ops.torch_ipex.pad_transform_fp16(
        query, key, value, heads, add_padding
    )


def pad_transform_bf16(query, key, value, heads, add_padding):
    return torch.ops.torch_ipex.pad_transform_bf16(
        query, key, value, heads, add_padding
    )


def allocate_workspace_fp32(
    hidden_dim,
    num_heads,
    prompt_length,
    batch_size,
    num_layers,
    mp_size,
    external_cache,
    rank,
    max_out_tokens,
    min_out_tokens,
):
    return torch.ops.torch_ipex.allocate_workspace_fp32(
        hidden_dim,
        num_heads,
        prompt_length,
        batch_size,
        num_layers,
        mp_size,
        external_cache,
        rank,
        max_out_tokens,
        min_out_tokens,
    )


def allocate_workspace_fp16(
    hidden_dim,
    num_heads,
    prompt_length,
    batch_size,
    num_layers,
    mp_size,
    external_cache,
    rank,
    max_out_tokens,
    min_out_tokens,
):
    return torch.ops.torch_ipex.allocate_workspace_fp16(
        hidden_dim,
        num_heads,
        prompt_length,
        batch_size,
        num_layers,
        mp_size,
        external_cache,
        rank,
        max_out_tokens,
        min_out_tokens,
    )


def allocate_workspace_bf16(
    hidden_dim,
    num_heads,
    prompt_length,
    batch_size,
    num_layers,
    mp_size,
    external_cache,
    rank,
    max_out_tokens,
    min_out_tokens,
):
    return torch.ops.torch_ipex.allocate_workspace_bf16(
        hidden_dim,
        num_heads,
        prompt_length,
        batch_size,
        num_layers,
        mp_size,
        external_cache,
        rank,
        max_out_tokens,
        min_out_tokens,
    )


def ti_dequantize_fp32(weight, qscale, groups):
    return torch.ops.torch_ipex.ti_dequantize_fp32(weight, qscale, groups)


def ti_dequantize_fp16(weight, qscale, groups):
    return torch.ops.torch_ipex.ti_dequantize_fp16(weight, qscale, groups)


def ti_dequantize_bf16(weight, qscale, groups):
    return torch.ops.torch_ipex.ti_dequantize_bf16(weight, qscale, groups)
