#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

at::Tensor bmm_impl(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& postop_tensors,
    const float dst_coeff = 1.0f);

} // namespace cpu
} // namespace torch_ipex
