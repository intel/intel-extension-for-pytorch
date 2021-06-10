#pragma once
#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>

namespace toolkit {
  std::vector<double> roc_auc_score(at::Tensor actual, at::Tensor predict);
  std::vector<double> roc_auc_score_all(at::Tensor actual, at::Tensor predict);
}
