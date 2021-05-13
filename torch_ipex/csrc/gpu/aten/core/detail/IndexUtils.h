#pragma once

#include <ATen/ATen.h>
#include <limits>


namespace xpu {
namespace dpcpp {
namespace detail {

bool maybeOverlappingIndices(const at::Tensor& t);
bool canUse32BitIndexMath(const at::Tensor &t, int64_t max_elem=std::numeric_limits<int32_t>::max());

} // detail
} // dpcpp
} // xpu
