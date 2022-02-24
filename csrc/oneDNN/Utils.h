#pragma once

#include <ATen/ATen.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <tensor/Context.h>

using namespace dnnl;

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

static inline dnnl::memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last = false) {
  switch (ndims) {
    case 1:
      return memory::format_tag::a;
    case 2:
      return memory::format_tag::ab;
    case 3:
      return is_channels_last ? memory::format_tag::acb
                              : memory::format_tag::abc;
    case 4:
      return is_channels_last ? memory::format_tag::acdb
                              : memory::format_tag::abcd;
    case 5:
      return is_channels_last ? memory::format_tag::acdeb
                              : memory::format_tag::abcde;
    case 6:
      return memory::format_tag::abcdef;
    case 7:
      return memory::format_tag::abcdefg;
    default:
      return memory::format_tag::any;
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

static inline memory::data_type get_onednn_dtype(const at::Tensor& tensor) {
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

static inline memory::dims get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

static inline bool eltwise_forward_valid(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    // return false if scalar_type not supported
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    case at::ScalarType::Half:
      break;
    case at::ScalarType::Int:
      break;
    case at::ScalarType::Char:
      break;
    case at::ScalarType::Byte:
      break;
    default:
      return false;
  };
  if (!at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor))
    return true;
  if (tensor.is_contiguous() || tensor.dim() == 1)
    return true;
  return false;
}

static inline bool eltwise_backward_valid(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    default:
      return false;
  };
  if (!at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor))
    return true;
  if (tensor.is_contiguous() || tensor.dim() == 1)
    return true;
  return false;
}

} // namespace oneDNN
} // namespace xpu
