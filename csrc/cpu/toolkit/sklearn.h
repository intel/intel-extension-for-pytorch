#pragma once
#include <ATen/Tensor.h>
#include <Macros.h>

#include <vector>

namespace toolkit {
IPEX_API std::vector<double> roc_auc_score(
    at::Tensor actual,
    at::Tensor predict);
IPEX_API std::vector<double> roc_auc_score_all(
    at::Tensor actual,
    at::Tensor predict);
} // namespace toolkit
