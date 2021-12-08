#pragma once

#include <ATen/Tensor.h>

#ifndef IS_CONTIGUOUS_ANY
#define IS_CONTIGUOUS_ANY(input_tensor)                             \
  input_tensor.is_contiguous(at::MemoryFormat::Contiguous) ||       \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) || \
      input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d)
#endif

namespace torch_ipex {

bool is_transposed_2d(const at::Tensor& tensor);

const char* scalarTypeName(const at::ScalarType type);
const char* LayoutName(const at::Layout layout);

} // namespace torch_ipex
