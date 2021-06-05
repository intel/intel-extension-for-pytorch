#pragma once

#include <ATen/ATen.h>
#include <oneapi/dnnl/dnnl.hpp>


using namespace dnnl;

namespace xpu {
namespace oneDNN {

static inline dnnl::memory::format_tag get_dnnl_default_format(int ndims) {
  switch (ndims) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    default:
      return dnnl::memory::format_tag::any;
  }
}

static bool is_supported_onednn_dtype(const at::Tensor& tensor) {
   switch (tensor.scalar_type()) {
   case at::ScalarType::Byte:
   case at::ScalarType::Char:
   case at::ScalarType::Int:
   case at::ScalarType::Half:
   case at::ScalarType::Float:
   case at::ScalarType::BFloat16:
   case at::ScalarType::QInt8:
   case at::ScalarType::QUInt8:
     return true;
   default:
     return false;
   };
}

static bool is_supported_dtype_in_binary_impl(at::ScalarType t) {
   switch (t) {
   case at::ScalarType::Byte:
   case at::ScalarType::Char:
   case at::ScalarType::Half:
   case at::ScalarType::Float:
     return true;
   default:
     return false;
   };
}

static bool is_supported_dtype_in_binary(at::ScalarType src0, at::ScalarType src1) {
  if (src0 == at::ScalarType::BFloat16 && src1 == at::ScalarType::BFloat16) {
    return true;
  } else if (is_supported_dtype_in_binary_impl(src0) &&
             is_supported_dtype_in_binary_impl(src1)) {
    return true;

  }
  return false;
}

static inline memory::data_type
get_onednn_dtype(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
  case at::ScalarType::Byte:
    return memory::data_type::u8;
  case at::ScalarType::Char:
    return memory::data_type::s8;
  case at::ScalarType::QInt8:
    return memory::data_type::s8;
  case at::ScalarType::QUInt8:
    return memory::data_type::u8;
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

}}
