#pragma once

#include <ATen/core/Tensor.h>
#include <tensor/Tensor.h>
#include <utils/DPCPP.h>

using namespace at;

namespace torch_ipex::xpu {
namespace dpcpp {
// Note: Only all input tensors are channels last format, cat's output would be
// channels last tensor. This may cause perf issue in the future.
inline c10::MemoryFormat cat_compute_output_memory_format(
    const MaterializedITensorListRef& tensors) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (auto mt : tensors) {
    auto t = mt.get();
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

inline bool is_channels_last(at::MemoryFormat fmt) {
  return (
      (at::MemoryFormat::ChannelsLast == fmt) ||
      (at::MemoryFormat::ChannelsLast3d == fmt));
}

inline bool is_smf_channels_last(const Tensor& t) {
  const auto ndim = t.ndimension();
  if (4 != ndim && 5 != ndim) {
    // channels last only supports 4D, 5D tensor
    return false;
  }

  return is_channels_last(t.suggest_memory_format());
}

inline MemoryFormat get_cl_tag_by_ndim(const int64_t ndim) {
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "ndim must be 3, 4 or 5 when get cl tag");
  if (3 == ndim) {
    // if doesn't enable channels last 1d
    return at::MemoryFormat::Contiguous;
  } else if (5 == ndim) {
    return at::MemoryFormat::ChannelsLast3d;
  } else {
    return at::MemoryFormat::ChannelsLast;
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
