#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "Reorder.h"
#include "Utils.h"

namespace xpu {
namespace oneDNN {

constexpr int src_batch_size_dim = 0;
constexpr int wgh_dst_channels_dim = 0;

static inline memory::format_tag conv_wgh_fmt(
    const int64_t ndim,
    const bool grouped = false,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::goiw : memory::format_tag::oiw)
        : (ndim == 4)
        ? (grouped ? memory::format_tag::goihw : memory::format_tag::oihw)
        : ((ndim == 5) ? (grouped ? memory::format_tag::goidhw
                                  : memory::format_tag::oidhw)
                       : memory::format_tag::undef);
  } else {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::gowi : memory::format_tag::owi)
        : (ndim == 4)
        ? (grouped ? memory::format_tag::gohwi : memory::format_tag::ohwi)
        : ((ndim == 5) ? (grouped ? memory::format_tag::godhwi
                                  : memory::format_tag::odhwi)
                       : memory::format_tag::undef);
  }
}

static inline memory::dims compatible_wgh_dims(
    const int64_t ndim,
    const int64_t groups,
    const int64_t oc,
    const int64_t ic,
    const IntArrayRef wsizes) {
  if (ndim == 3) {
    auto kw = wsizes[2];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kw})
                         : memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

static inline memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? memory::format_tag::ncw
        : ((ndim == 4) ? memory::format_tag::nchw
                       : ((ndim == 5) ? memory::format_tag::ncdhw
                                      : memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? memory::format_tag::nwc
        : ((ndim == 4) ? memory::format_tag::nhwc
                       : ((ndim == 5) ? memory::format_tag::ndhwc
                                      : memory::format_tag::undef));
  }
}

static inline memory::dims conv_dst_tz(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  memory::dims dst_tz(ndim);
  dst_tz[0] = src_tz[src_batch_size_dim];
  dst_tz[1] = wgh_tz[wgh_dst_channels_dim];
  for (size_t d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (wgh_tz[d] - 1) + 1;
    dst_tz[d] =
        (src_tz[d] +
         (padding_front_top_left[d - 2] + padding_back_bottom_right[d - 2]) -
         kernel) /
            stride[d - 2] +
        1;
  }
  return dst_tz;
}

static inline memory::dims compatible_dilation(IntArrayRef& dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

} // namespace oneDNN
} // namespace xpu
