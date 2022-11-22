#pragma once

#include <ATen/core/Tensor.h>
#include <tensor/Tensor.h>
#include <utils/DPCPP.h>

// Channels last 1D is only supported by IPEX GPU
#define CL1D_OFFSET 20
#define CHANNELSLAST1D_DPCPP \
  ((at::MemoryFormat)((int)at::MemoryFormat::ChannelsLast3d + CL1D_OFFSET))

using namespace at;

namespace xpu {
namespace dpcpp {

inline std::vector<int64_t> get_channels_last_strides_1d_dpcpp(
    IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  switch (sizes.size()) {
    case 3:
      strides[1] = 1;
      strides[2] = sizes[1];
      strides[0] = strides[2] * sizes[2];
      return strides;
    case 2:
      strides[0] = 1;
      strides[1] = sizes[0];
      return strides;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast1d doesn't support size ", sizes.size());
  }
}

inline bool is_channels_last_strides_1d_s3_dpcpp(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  int64_t min = 0;
  // special case for trivial C dimension. default to NCL
  if (strides[1] == 0) {
    return false;
  }
  // loop strides indices
  for (auto& d : {1, 2, 0}) {
    if (sizes[d] == 0) {
      return false;
    }
    if (strides[d] < min) {
      return false;
    }
    // Fallback to NCL as default layout for ambiguous cases
    // This is the flaw of implicit memory_format from strides.
    // N11 tensor with identical strides for size 1 dimension;
    // Two cases could lead us here:
    // a. N11 contiguous Tensor ([N,1,1]@[1,1,1])
    // b. N1L contiguous Tensor sliced on the L-dimension. ([N,1,1]@[L,L,L])
    if (d == 0 && min == strides[1]) {
      return false;
    }

    min = strides[d];
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  return true;
}

inline bool is_channels_last_strides_1d_dpcpp(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  switch (sizes.size()) {
    case 3:
      return is_channels_last_strides_1d_s3_dpcpp(sizes, strides);
    case 2:
      // TODO dim == 2 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

inline bool compute_strides_like_channels_last_1d_dpcpp(const Tensor& t) {
  return is_channels_last_strides_1d_dpcpp(t.sizes(), t.strides());
}

inline at::MemoryFormat suggest_memory_format_dpcpp(
    const Tensor& t,
    bool channels_last_strides_exact_match = false) {
  const auto ndim = t.ndimension();
  if (3 != ndim) {
    return t.suggest_memory_format();
  }

  // ipex gpu channels last 1d
  if (!t.is_mkldnn() && !t.is_sparse()) {
    if (compute_strides_like_channels_last_1d_dpcpp(t)) {
      if (!channels_last_strides_exact_match ||
          get_channels_last_strides_1d_dpcpp(t.sizes()) == t.strides()) {
        return CHANNELSLAST1D_DPCPP;
      }
    }
  }

  return at::MemoryFormat::Contiguous;
}

// Note: Only all input tensors are channels last format, cat's output would be
// channels last tensor. This may cause perf issue in the future.
inline c10::MemoryFormat cat_compute_output_memory_format(
    const MaterializedITensorListRef& tensors,
    bool channels_last_strides_exact_match = false) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (auto mt : tensors) {
    auto t = mt.get();
    auto f = suggest_memory_format_dpcpp(t, channels_last_strides_exact_match);
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

inline Tensor& convert_tensor_to_channels_last_1d(Tensor& t) {
  const auto ndim = t.ndimension();
  if (3 != ndim) {
    return t;
  }

  if (1 == t.size(0)) {
    t = t.transpose(1, -1).contiguous().transpose(1, -1);
    return t;
  }

  if (1 == t.size(1)) {
    t = t.as_strided(t.sizes(), {t.size(1) * t.size(-1), 1, t.size(1)});
    return t;
  }

  t = t.view({t.size(0), t.size(1), 1, t.size(2)});
  t = t.contiguous(at::MemoryFormat::ChannelsLast);
  t = t.view({t.size(0), t.size(1), t.size(3)});
  return t;
}

inline Tensor empty_like_dpcpp(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto ndim = self.ndimension();
  if (3 != ndim || CHANNELSLAST1D_DPCPP != optional_memory_format) {
    return empty_like(self, options, optional_memory_format);
  }
  // ipex gpu channels last 1d
  Tensor tmp = empty_like(self);
  return convert_tensor_to_channels_last_1d(tmp);
}

inline Tensor empty_opaque_tensor_dpcpp(
    at::AtenIpexTypeXPU::DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto ndim = meta.dims().size();
  if (3 != ndim || CHANNELSLAST1D_DPCPP != optional_memory_format) {
    return at::AtenIpexTypeXPU::empty_opaque_tensor(
        meta, options, optional_memory_format);
  }
  // ipex gpu channels last 1d
  Tensor tmp =
      at::AtenIpexTypeXPU::empty_opaque_tensor(meta, options, c10::nullopt);
  return convert_tensor_to_channels_last_1d(tmp);
}

inline bool is_channels_last(at::MemoryFormat fmt) {
  if (
#ifdef USE_CHANNELS_LAST_1D
      CHANNELSLAST1D_DPCPP == fmt ||
#endif
      at::MemoryFormat::ChannelsLast == fmt ||
      at::MemoryFormat::ChannelsLast3d == fmt) {
    return true;
  }
  return false;
}

inline bool is_smf_channels_last(const Tensor& t) {
  const auto ndim = t.ndimension();
  if (
#ifdef USE_CHANNELS_LAST_1D
      3 != ndim &&
#endif
      4 != ndim && 5 != ndim) {
    // channels last only supports 3D, 4D, 5D tensor
    return false;
  }

  return is_channels_last(suggest_memory_format_dpcpp(t));
}

inline MemoryFormat get_cl_tag_by_ndim(const int64_t ndim) {
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "ndim must be 3, 4 or 5 when get cl tag");
  if (3 == ndim) {
#ifdef USE_CHANNELS_LAST_1D
    return CHANNELSLAST1D_DPCPP;
#else
    // if doesn't enable channels last 1d
    return at::MemoryFormat::Contiguous;
#endif
  } else if (5 == ndim) {
    return at::MemoryFormat::ChannelsLast3d;
  } else {
    return at::MemoryFormat::ChannelsLast;
  }
}

} // namespace dpcpp
} // namespace xpu
