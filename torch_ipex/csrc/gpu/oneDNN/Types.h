#pragma once

#define ONEDNN_SCALES_MASK_BY_CHANNEL(x) (1 << x)

namespace xpu {
namespace oneDNN {

enum post_attr {
  with_relu = 0b01,
  with_sum = 0b10,
  with_sigmoid = 0b100,
};

}}
