#pragma once

#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace xpu {

namespace {
/* IPEX_GENERAL_CONV_SYMBOL_DECLARATION
This macro is used to convinently delcare the necessary Symbols which will be
used in op fusion, all the Symbol defined by this macro will be in form like:

static auto conv2d_op_sym = Symbol::fromQualString("xpu::conv2d_op")
static auto _convolution_op_sym = Symbol::fromQualString("xpu::_convolution_op")
static auto q_conv2d_op_sym = Symbol::fromQualString("xpu::q_conv2d_op")

and those defination above can be adopted by macro IPEX_DEFINE_CONV_FUSION to
generate the rule of pots-op fusion.
*/
#define IPEX_QCONV_SYMBOL_DECLARATION(func) \
  static auto q_conv2d_##func##_sym =       \
      Symbol::fromQualString("xpu::q_conv2d_" #func)

#define IPEX_CONV_SYMBOL_DECLARATION(func) \
  static auto conv2d_##func##_sym = Symbol::fromQualString("xpu::conv2d_" #func)

#define IPEX__CONV_SYMBOL_DECLARATION(func) \
  static auto _convolution_##func##_sym =   \
      Symbol::fromQualString("xpu::_convolution_" #func)

#define IPEX_GENERAL_CONV_SYMBOL_DECLARATION(func) \
  IPEX_QCONV_SYMBOL_DECLARATION(func);             \
  IPEX_CONV_SYMBOL_DECLARATION(func);              \
  IPEX__CONV_SYMBOL_DECLARATION(func);
} // namespace

IPEX_GENERAL_CONV_SYMBOL_DECLARATION(sqrt);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(tanh);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(square);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(abs);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(exp);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(log);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(round);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(log_sigmoid);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(hardswish);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(mish);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(silu);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(gelu);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(hardsigmoid);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(elu);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(pow);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(hardtanh);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(sigmoid);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(leaky_relu);
IPEX_GENERAL_CONV_SYMBOL_DECLARATION(relu);

static auto _conv_sym = Symbol::fromQualString("aten::_convolution");
static auto reorder_sym = Symbol::fromQualString("xpu::reorder");
static auto batch_norm_sym = Symbol::fromQualString("xpu::batch_norm");
static auto pad_conv2d_sym = Symbol::fromQualString("xpu::pad_conv2d");
static auto conv2d_sum_sym = Symbol::fromQualString("xpu::conv2d_sum");
static auto conv2d_relu_sum_sym =
    Symbol::fromQualString("xpu::conv2d_relu_sum");
static auto _convolution_sum_sym =
    Symbol::fromQualString("xpu::_convolution_sum");
static auto _convolution_sum_relu_sym =
    Symbol::fromQualString("xpu::_convolution_sum_relu");
static auto conv2d_sum_relu_sym =
    Symbol::fromQualString("xpu::conv2d_sum_relu");
static auto matmul_add_sym = Symbol::fromQualString("xpu::matmul_add");
static auto t_matmul_sym = Symbol::fromQualString("xpu::t_matmul");
static auto trans_matmul_sym = Symbol::fromQualString("xpu::trans_matmul");
static auto t_matmul_add_sym = Symbol::fromQualString("xpu::t_matmul_add");
static auto t_matmul_add_add_sym =
    Symbol::fromQualString("xpu::t_matmul_add_add");
static auto t_matmul_add_gelu_sym =
    Symbol::fromQualString("xpu::t_matmul_add_gelu");
static auto trans_matmul_div_sym =
    Symbol::fromQualString("xpu::trans_matmul_div");
static auto mul_add_sym = Symbol::fromQualString("xpu::mul_add");
static auto q_conv2d_sum_relu_sym =
    Symbol::fromQualString("xpu::q_conv2d_sum_relu");
static auto q_conv2d_dequantize_sym =
    Symbol::fromQualString("xpu::q_conv2d_dequantize");
static auto softplus_tanh_sym = Symbol::fromQualString("xpu::softplus_tanh");
static auto softplus_tanh_mul_sym =
    Symbol::fromQualString("xpu::softplus_tanh_mul");
static auto q_conv2d_dequantize_softplus_tanh_mul_sym =
    Symbol::fromQualString("xpu::q_conv2d_dequantize_softplus_tanh_mul");
static auto q_conv2d_sym = Symbol::fromQualString("quantized::conv2d");
static auto q_conv2d_dequantize_softplus_tanh_mul_quantize_sym =
    Symbol::fromQualString(
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize");
static auto q_conv2d_dequantize_softplus_tanh_mul_quantize_add_sym =
    Symbol::fromQualString(
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize_add");
static auto linear_gelu_sym = Symbol::fromQualString("xpu::linear_gelu");
static auto linear_relu_sym = Symbol::fromQualString("xpu::linear_relu");
static auto linear_sigmoid_sym = Symbol::fromQualString("xpu::linear_sigmoid");
static auto linear_add_sym = Symbol::fromQualString("xpu::linear_add");
static auto dequant_pixelshuffle_sym =
    Symbol::fromQualString("xpu::dequant_pixelshuffle");
static auto dequant_pixelshuffle_quant_sym =
    Symbol::fromQualString("xpu::dequant_pixelshuffle_quant");
static auto permute_contiguous_sym =
    Symbol::fromQualString("xpu::permute_contiguous");
static auto convolution_silu_sym =
    Symbol::fromQualString("xpu::_convolution_silu");
static auto conv2d_binary_mul_sym =
    Symbol::fromQualString("xpu::conv2d_binary_mul");

// Fold weights of batch_norm with conv2d's
static auto fold_weight_sym = Symbol::fromQualString("xpu::fold_weight");
static auto fold_bias_sym = Symbol::fromQualString("xpu::fold_bias");
} // namespace xpu

#if 0
Operation createDNNL_reorder(const Node *node);
Operation createDNNL_relu(const Node *node);
Operation createDNNL_relu_(const Node *node);
Operation createDNNL_conv2d(const Node *node);
Operation createDNNL_conv2d_relu(const Node *node);
Operation createDNNL_batch_norm(const Node *node);
Operation createDNNL_fold_weight(const Node *node);
Operation createDNNL_fold_bias(const Node *node);
Operation createDNNL_pooling_max_2d(const Node *node);
Operation createDNNL_pooling_avg_2d(const Node *node);
Operation createDNNL_sum(const Node* node);
Operation createDNNL_sum_(const Node *node);
Operation createDNNL_conv2d_sum(const Node *node);
Operation createDNNL_conv2d_sum_relu(const Node* node);
#endif
} // namespace jit
} // namespace torch
