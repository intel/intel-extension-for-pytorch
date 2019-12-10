#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUDefault {
public:
  static at::Tensor add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor ones(at::IntArrayRef size, const at::TensorOptions & options);
  static at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format);
  static at::Tensor view(const at::Tensor& self, at::IntArrayRef size);
  static at::Tensor to(const at::Tensor & self, c10::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format);
};

} // cpu
} // torch_ipex
