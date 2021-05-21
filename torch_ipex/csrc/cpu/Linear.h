#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

at::Tensor linear_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr);

}  // namespace cpu
}  // namespace torch_ipex
