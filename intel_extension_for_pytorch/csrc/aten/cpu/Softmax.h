#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor softmax_impl(const at::Tensor& input, const int64_t dim);
} // namespace cpu
} // namespace torch_ipex
