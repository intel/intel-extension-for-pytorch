#pragma once

#include <ATen/ATen.h>
#include <dnnl.hpp>

using namespace dnnl;

namespace at {
namespace dpcpp {
namespace oneDNN {

static bool is_supported_onednn_dtype(const at::Tensor& tensor) {
   switch (tensor.scalar_type()) {
   case at::ScalarType::Byte:
   case at::ScalarType::Char:
   case at::ScalarType::Int:
   case at::ScalarType::Half:
   case at::ScalarType::Float:
   case at::ScalarType::BFloat16:
     return true;
   default:
     return false;
   };
}

static inline memory::data_type
get_onednn_dtype(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
  case at::ScalarType::Byte:
    return memory::data_type::u8;
  case at::ScalarType::Char:
    return memory::data_type::s8;
  case at::ScalarType::Int:
    return memory::data_type::s32;
  case at::ScalarType::Half:
    return memory::data_type::f16;
  case at::ScalarType::Float:
    return memory::data_type::f32;
  case at::ScalarType::BFloat16:
    return memory::data_type::bf16;
  default:
    return memory::data_type::undef;
  };
}

static inline memory::dims
get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims
get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

}}}
