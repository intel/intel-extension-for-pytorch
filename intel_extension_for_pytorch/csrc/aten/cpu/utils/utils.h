#pragma once

#include <ATen/Tensor.h>

#ifndef IS_CONTIGUOUS_ANY
#define IS_CONTIGUOUS_ANY(input_tensor)                             \
  input_tensor.is_contiguous(at::MemoryFormat::Contiguous) ||       \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) || \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d)
#endif