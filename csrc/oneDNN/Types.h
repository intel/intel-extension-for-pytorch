#pragma once

#define ONEDNN_SCALES_MASK_BY_CHANNEL(x) (1 << x)

namespace xpu {
namespace oneDNN {

enum post_attr {
  with_relu = 0b01,
  with_sum = 0b10,
  with_sigmoid = 0b100,
  with_bin_mul = 0b1000,
  with_bin_add = 0b10000,
  with_bin_sub = 0b100000,
  with_gelu = 0b1000000,

};

}
} // namespace xpu
