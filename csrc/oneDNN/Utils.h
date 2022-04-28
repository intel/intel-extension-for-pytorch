#pragma once

#include <ATen/ATen.h>
#include <core/MemoryFormat.h>
#include <core/detail/TensorInfo.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <tensor/Context.h>
#include <utils/Macros.h>

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
    bool is_channels_last = false,
    bool allow_undef = false) {
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
    case 8:
      return memory::format_tag::abcdefgh;
    case 9:
      return memory::format_tag::abcdefghi;
    case 10:
      return memory::format_tag::abcdefghij;
    case 11:
      return memory::format_tag::abcdefghijk;
    case 12:
      return memory::format_tag::abcdefghijkl;
    default:
      if (!allow_undef) {
        TORCH_CHECK(false, "oneDNN doesn't support tensor dimension > 12");
      }
      return memory::format_tag::undef;
  }
}

static inline memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef = false) {
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
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(tensor.scalar_type()),
            " is not supported in oneDNN!");
      }
      return memory::data_type::undef;
  };
}

static bool is_supported_onednn_dtype(const at::Tensor& tensor) {
  return get_onednn_dtype(tensor, /*allow_undef*/ true) ==
          memory::data_type::undef
      ? false
      : true;
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

static inline std::vector<int64_t> compatible_groups_conv_strides(
    const at::Tensor& wgh,
    const at::Tensor& wgh_) {
  std::vector<int64_t> strides(wgh_.sizes().size());
  strides[4] = wgh.strides()[3];
  strides[3] = wgh.strides()[2];
  strides[2] = wgh.strides()[1];
  strides[1] = wgh.strides()[0];
  strides[0] = wgh_.sizes()[1] * wgh.strides()[0];
  return strides;
}

static inline bool is_onednn_layout(const at::Tensor& tensor) {
  return !at::AtenIpexTypeXPU::DPCPPTensorContext::is_plain(tensor);
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

static bool is_wrapped_number(const Tensor& t) {
  return t.unsafeGetTensorImpl()->is_wrapped_number();
}

static inline bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other) {
  return (
      self.sizes() != other.sizes() &&
      is_expandable_to(other.sizes(), self.sizes()));
}

static inline bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other) {
  // FIXME: update onednn
  if (self.sizes() != other.sizes() &&
      !is_broadcast_from_other_to_self(self, other))
    return false;

  /* If the following conditions are satisfied, then oneDNN path will be
     selected:
     * 1. self and other should be xpu tensor and be defined.
     * 2. self or other should not be scalar (wrapped tensor).
     * 3. dim of self and other should be equal and must be larger than 0.
     * 4. the datatype should be supported by oneDNN primitive.
     * 5. self and other should be in the same datatype.
     * 6. self and other should be contiguous or channel-last contiguous.*/

  using namespace at::AtenIpexTypeXPU;

  // 1. self and other should be xpu tensor and be defined.
  if ((!self.defined()) || (!other.defined()) || (!self.is_xpu()) ||
      (!other.is_xpu()))
    return false;

  // 2. self or other should not be scalar (wrapped tensor).
  if (is_wrapped_number(self) || is_wrapped_number(other))
    return false;

  // 3. dim of self and other should be equal and must be larger than 0.
  if ((self.dim() <= 0) || (other.dim() <= 0) || (self.dim() != other.dim()))
    return false;

  // 4. the datatype should be supported by oneDNN primitive.
  switch (self.scalar_type()) {
    case at::ScalarType::Char:
      break;
    case at::ScalarType::Byte:
      break;
    case at::ScalarType::Half:
      break;
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    default:
      return false;
  };

  // 5. self and other should be in the same datatype.
  if (self.scalar_type() != other.scalar_type())
    return false;

  // 6. self and other should be contiguous or channel-last contiguous.
  const auto ndim = self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }
  if ((self.is_contiguous() && other.is_contiguous()) ||
      (self.is_contiguous(cl_tag) && other.is_contiguous(cl_tag)))
    return true;
  return false;
}

static inline bool softmax_valid(const at::Tensor& self) {
  if (!self.is_contiguous())
    return false;

  if (self.sizes().size() > 4 || self.sizes().size() < 1)
    return false;

  // the datatype should be supported by oneDNN primitive.
  switch (self.scalar_type()) {
    case at::ScalarType::Half:
      break;
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    default:
      return false;
  };
  return true;
}

static inline bool softmax_backward_valid(
    const at::Tensor& grad,
    const at::Tensor& output,
    const at::Tensor& input) {
  if (!grad.is_contiguous() || !output.is_contiguous())
    return false;

  if (input.sizes().size() > 4 || input.sizes().size() < 1)
    return false;

  // the datatype should be supported by oneDNN primitive.
  switch (input.scalar_type()) {
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    default:
      return false;
  };
  return true;
}

static inline bool cat_valid(const TensorList& tensors) {
  for (int i = 0; i < tensors.size(); i++) {
    const Tensor& tensor = tensors[i];
    if (tensor.defined()) {
      if (tensor.scalar_type() == ScalarType::Bool ||
          tensor.scalar_type() == ScalarType::Short ||
          tensor.scalar_type() == ScalarType::Double ||
          tensor.scalar_type() == ScalarType::Long ||
          tensor.scalar_type() == ScalarType::ComplexFloat ||
          tensor.scalar_type() == ScalarType::ComplexDouble ||
          tensor.dim() > MAX_TENSORINFO_DIMS) {
        return false;
      }
    }
  }
  return true;
}

} // namespace oneDNN
} // namespace xpu
