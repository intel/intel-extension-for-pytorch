import transformer_inference

def gated_activation(activation, bias, actFun):
    return transformer_inference.gated_activation(activation, bias, actFun)

def bias_gelu_fp32(vals, bias):
    return transformer_inference.bias_gelu_fp32(vals, bias)

def layer_norm(input, gamma, beta, epsilon):
    return transformer_inference.layer_norm(input, gamma, beta, epsilon)

def _layer_norm_residual(input, bias, residual, gamma, beta, epsilon):
    return transformer_inference._layer_norm_residual(input, bias, residual, gamma, beta, epsilon)

def layer_norm_residual_store_pre_ln_res(input, bias, residual, gamma, beta, epsilon):
    return transformer_inference.layer_norm_residual_store_pre_ln_res(input, bias, residual, gamma, beta, epsilon)

def rms_norm(input, gamma, epsilon):
    return transformer_inference.rms_norm(input, gamma, epsilon)

def pre_rms_norm(input, residual, gamma, epsilon):
    return transformer_inference.pre_rms_norm(input, residual, gamma, epsilon)

def moe_res_matmul(moe_res, coef, output):
    return transformer_inference.moe_res_matmul(moe_res, coef, output)

def residual_add_bias_fp16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln):
    return transformer_inference.residual_add_bias_fp16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln)

def residual_add_bias_fp32(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln):
    return transformer_inference.residual_add_bias_fp32(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln)

def residual_add_bias_bf16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln):
    return transformer_inference.residual_add_bias_bf16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, preln)

def bias_gelu_fp16(activations, bias):
    return transformer_inference.bias_gelu_fp16(activations, bias)

def bias_gelu_fp32(activations, bias):
    return transformer_inference.bias_gelu_fp32(activations, bias)

def bias_add_fp16(activations, bias):
    return transformer_inference.bias_add_fp16(activations, bias)

def bias_add_fp32(activations, bias):
    return transformer_inference.bias_add_fp32(activations, bias)

def bias_add_bf16(activations, bias):
    return transformer_inference.bias_add_bf16(activations, bias)

def bias_relu_fp16(activations, bias):
    return transformer_inference.bias_relu_fp16(activations, bias)

def bias_relu_fp32(activations, bias):
    return transformer_inference.bias_relu_fp32(activations, bias)

def bias_relu_bf16(activations, bias):
    return transformer_inference.bias_relu_bf16(activations, bias)

def bias_gelu_fp16(activations, bias):
    return transformer_inference.bias_gelu_fp16(activations, bias)

def bias_gelu_fp32(activations, bias):
    return transformer_inference.bias_gelu_fp32(activations, bias)

def bias_gelu_bf16(activations, bias):
    return transformer_inference.bias_gelu_bf16(activations, bias)