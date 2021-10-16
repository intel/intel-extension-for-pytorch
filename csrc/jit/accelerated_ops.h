#pragma once

#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace xpu {
static auto reorder_sym = Symbol::fromQualString("xpu::reorder");
static auto batch_norm_sym = Symbol::fromQualString("xpu::batch_norm");
static auto conv2d_relu_sym = Symbol::fromQualString("xpu::conv2d_relu");
static auto conv2d_sum_sym = Symbol::fromQualString("xpu::conv2d_sum");
static auto conv2d_relu_sum_sym =
    Symbol::fromQualString("xpu::conv2d_relu_sum");
static auto conv2d_sum_relu_sym =
    Symbol::fromQualString("xpu::conv2d_sum_relu");
static auto conv2d_sigmoid_sym = Symbol::fromQualString("xpu::conv2d_sigmoid");
static auto matmul_add_sym = Symbol::fromQualString("xpu::matmul_add");
static auto t_matmul_sym = Symbol::fromQualString("xpu::t_matmul");
static auto trans_matmul_sym = Symbol::fromQualString("xpu::trans_matmul");
static auto t_matmul_add_sym = Symbol::fromQualString("xpu::t_matmul_add");
static auto t_matmul_add_dropout_sym =
    Symbol::fromQualString("xpu::t_matmul_add_dropout");
static auto t_matmul_add_add_sym =
    Symbol::fromQualString("xpu::t_matmul_add_add");
static auto t_matmul_add_gelu_sym =
    Symbol::fromQualString("xpu::t_matmul_add_gelu");
static auto trans_matmul_div_sym =
    Symbol::fromQualString("xpu::trans_matmul_div");
static auto trans_matmul_scale_add_sym =
    Symbol::fromQualString("xpu::trans_matmul_scale_add");
static auto mul_add_sym = Symbol::fromQualString("xpu::mul_add");
static auto q_conv2d_sum_relu_sym =
    Symbol::fromQualString("xpu::q_conv2d_sum_relu");
static auto t_addmm_sym = Symbol::fromQualString("xpu::t_addmm");
static auto t_addmm_dropout_sym =
    Symbol::fromQualString("xpu::t_addmm_dropout");
static auto t_addmm_relu_sym = Symbol::fromQualString("xpu::t_addmm_relu");
static auto t_addmm_sigmoid_sym =
    Symbol::fromQualString("xpu::t_addmm_sigmoid");
static auto dequant_pixelshuffle_sym =
    Symbol::fromQualString("xpu::dequant_pixelshuffle");
static auto dequant_pixelshuffle_quant_sym =
    Symbol::fromQualString("xpu::dequant_pixelshuffle_quant");

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
