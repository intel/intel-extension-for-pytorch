#pragma once

#include <ATen/Tensor.h>

// NOTE:
// Below are Helper functions for is_channels_last_strides_xd.
// 1. Please do not combine these helper functions, each helper function handles
// exactly one case of sizes + memory_format, by doing this, the strides indices
// will be a constant array and we can access it using constant index number,
// the compiler will fully unroll the loop on strides indices to gain a better
// performance.
// 2. No error check in helper function, caller ensures the correctness of the
// input
// 3. All helper functions have similar comments, only 1st helper function is
// commented here.

// The following code is updated from
// https://github.com/pytorch/pytorch/blob/0a66d5b3253fd2d2304f3897526db3c8fb139376/c10/core/MemoryFormat.h#L116
inline bool is_channels_last_1d(const at::Tensor& input) {
  if (input.dim() != 3) {
    return false;
  }
  auto sizes = input.sizes();
  auto strides = input.strides();
  int64_t min = 0;
  if (strides[1] == 0) {
    return false;
  }
  for (auto& d : {1, 2, 0}) {
    if (sizes[d] == 0) {
      return false;
    }
    if (strides[d] < min) {
      return false;
    }
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

#ifndef IS_CONTIGUOUS_ANY
#define IS_CONTIGUOUS_ANY(input_tensor)                               \
  input_tensor.is_contiguous(at::MemoryFormat::Contiguous) ||         \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||   \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d) || \
      is_channels_last_1d(input_tensor)
#endif
