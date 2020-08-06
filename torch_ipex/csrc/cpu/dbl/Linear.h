#pragma once

#include <ATen/Tensor.h>

#include "cpu/dil/dil.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace linear {

dil::tensor linear_impl(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    const dil::scale_t& dst_scales = dil::scale_t(),
    const dil::attr_t& attr = dil::attr_t());

} // namespace linear
} // namespace dbl
} // namespace cpu
} // namespace torch_ipex
