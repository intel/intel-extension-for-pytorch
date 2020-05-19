#pragma once

#include <ideep.hpp>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch { namespace jit {
// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace dnnl {
  static auto reorder = Symbol::fromQualString("dnnl::reorder");
  static auto conv2d = Symbol::fromQualString("dnnl::conv2d");
  static auto relu = Symbol::fromQualString("dnnl::relu");
  static auto relu_ = Symbol::fromQualString("dnnl::relu_");
  static auto batch_norm = Symbol::fromQualString("dnnl::batch_norm");
  static auto conv2d_relu = Symbol::fromQualString("dnnl::conv2d_relu");
  static auto pooling_max_2d = Symbol::fromQualString("dnnl::pooling_max_2d");
  static auto pooling_avg_2d = Symbol::fromQualString("dnnl::pooling_avg_2d");
  static auto sum = Symbol::fromQualString("dnnl::sum");
  static auto sum_ = Symbol::fromQualString("dnnl::sum_");
  static auto conv2d_sum = Symbol::fromQualString("dnnl::conv2d_sum");
  static auto conv2d_relu_sum = Symbol::fromQualString("dnnl::conv2d_relu_sum");
  static auto conv2d_sum_relu = Symbol::fromQualString("dnnl::conv2d_sum_relu");

  // Fold weights of batch_norm with conv2d's
  static auto fold_weight =
    Symbol::fromQualString("dnnl::fold_weight");
  static auto fold_bias =
    Symbol::fromQualString("dnnl::fold_bias");
}

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
}} // namespace torch::jit
