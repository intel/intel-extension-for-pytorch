#pragma once

#include <ATen/ATen.h>
#include <core/MemoryFormat.h>
#include <core/detail/TensorInfo.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/Macros.h>
#include <utils/Settings.h>

using namespace dnnl;

// FIXME: In some cases, for example, concat, reorder, and etc.
// oneDNN only supports dims <= 6 for now.
#define MAX_ONEDNN_SUPPORTED_DIMS 6
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
  with_mish = 0b10000000,
  with_linear = 0b100000000,
};

static inline memory::format_tag get_dnnl_default_format(
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

static inline memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,
    bool allow_undef = false) {
  if (tensor.scalar_type() == at::ScalarType::Double)
    return memory::data_type::f64;
  return get_onednn_dtype(tensor, allow_undef);
}

static bool is_supported_onednn_dtype(const at::Tensor& tensor) {
  return get_onednn_dtype(tensor, /*allow_undef*/ true) ==
          memory::data_type::undef
      ? false
      : true;
}

static inline fpmath_mode get_onednn_fpmath_mode() {
  auto math_mode = Settings::I().get_fp32_math_mode();
  switch (math_mode) {
    case FP32_MATH_MODE::TF32:
      return fpmath_mode::tf32;
    case FP32_MATH_MODE::BF32:
      return fpmath_mode::bf16;
    default: // use FP32_MATH_MODE::FP32 as default
      return fpmath_mode::strict;
  }
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

static inline memory::desc get_onednn_md(const at::Tensor& tensor) {
  return {
      get_onednn_dims(tensor),
      get_onednn_dtype(tensor),
      get_onednn_strides(tensor)};
}

template <typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
  for (size_t i = 0; i < size; ++i)
    dst[i] = src[i];
}

inline bool onednn_strides_check(const Tensor& src) {
  auto adims = xpu::oneDNN::get_onednn_dims(src);
  int ndims = (int)adims.size();
  auto dims = adims.data();
  auto data_type = static_cast<dnnl_data_type_t>(
      xpu::oneDNN::get_onednn_dtype(src, /*allow_undef*/ true));
  auto strides_info = xpu::oneDNN::get_onednn_strides(src);
  auto strides = strides_info.empty() ? nullptr : &strides_info[0];

  auto md = dnnl_memory_desc_t();
  md.ndims = ndims;
  array_copy(md.dims, dims, ndims);
  md.data_type = data_type;
  array_copy(md.padded_dims, dims, ndims);
  md.format_kind = dnnl_format_kind_t::dnnl_blocked;
  if (strides == nullptr || md.ndims == 0 ||
      md.format_kind != dnnl_format_kind_t::dnnl_blocked)
    return true;

  dnnl_dims_t blocks = {0};
  int perm[DNNL_MAX_NDIMS] = {0};
  for (int d = 0; d < md.ndims; ++d) {
    // no strides check needed for empty tensor
    if (md.padded_dims[d] == 0)
      return true;

    // no strides verification for runtime dims
    if (strides[d] == DNNL_RUNTIME_DIM_VAL)
      return true;

    perm[d] = d;
    blocks[d] = 1;
  }

  auto block_size = 1;
  const auto& blk = md.format_desc.blocking;
  for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
    blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
    block_size *= blk.inner_blks[iblk];
  }

  // A custom comparator to yield linear order on perm
  auto idx_sorter = [&](const int a, const int b) -> bool {
    if (strides[a] == strides[b] && md.padded_dims[a] == md.padded_dims[b])
      return a < b;
    else if (strides[a] == strides[b])
      return md.padded_dims[a] < md.padded_dims[b];
    else
      return strides[a] < strides[b];
  };
  std::sort(perm, perm + md.ndims, idx_sorter);

  auto min_stride = block_size;
  for (int idx = 0; idx < md.ndims; ++idx) {
    const int d = perm[idx];

    // Make an exception for strides[d] == 0 as it has broadcast semantics
    // Note: owing to being sorted, these are the initial strides
    if (strides[d] == 0)
      continue;
    else if (strides[d] < min_stride)
      return false;

    // update min_stride for next iteration
    const auto padded_dim = md.padded_dims[d];
    min_stride = block_size * strides[d] * (padded_dim / blocks[d]);
  }
  return true;
}

static inline bool is_broadcast(const at::Tensor& t) {
  for (int i = 0; i < t.dim(); i++) {
    if (t.stride(i) == 0)
      return true;
  }
  return false;
}

static inline bool is_onednn_matmul_strides(
    const at::Tensor& tensor,
    bool is_dst = false) {
  // https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html
  // oneDNN matmul only support 2-dim and 3-dim
  // 2D src(Mxk), wei(KxN), dst(MxN)
  // 3D src(SxMxK), wei(WxKxN), dst(DxMxN)
  auto sizes = tensor.sizes();
  auto tensor_dim = sizes.size();
  if (tensor_dim != 2 && tensor_dim != 3)
    return false;

  // the overlaped cases are not supported
  memory::dims strides = get_onednn_strides(tensor);
  int64_t storage_size = 1;
  for (size_t dim = 0; dim < tensor_dim; ++dim)
    storage_size += (sizes[dim] - 1) * strides[dim];
  if (storage_size < tensor.numel())
    return false;

  // the broadcast cases are not supported
  if (is_broadcast(tensor)) {
    return false;
  }

  if (is_dst) {
    // The memory format of the destination tensor should always
    // be plain with n axis contiguous
    if (strides[-1] != 1)
      return false;
  } else {
    // the src and weight must have at least one of the axes
    // m or k and n or k contiguous (i.e., stride=1) respectively.
    if (strides[tensor_dim - 1] != 1 && strides[tensor_dim - 2] != 1)
      return false;
  }

  if (!onednn_strides_check(tensor))
    return false;

  return true;
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
          tensor.dim() > MAX_ONEDNN_SUPPORTED_DIMS) {
        return false;
      }
    }
  }
  return true;
}

// judge to use block or not for Conv
static inline bool use_blocked_format_for_conv(const at::Tensor& src) {
  if (!src.defined() || src.is_sparse()) {
    // suggest plain
    return false;
  }

  if (Settings::I().is_onednn_layout_enabled()) {
    // suggest block
    return true;
  }

  // inference workloads on ATSM platform, the conv will use blocked format
  // used double support to distinguish is atsm or not
  auto is_auto_transpose = !dpcppSupportFP64();
  auto suggest_weight_block = is_auto_transpose &&
      (c10::InferenceMode::is_enabled() || !at::GradMode::is_enabled()) &&
      !is_smf_channels_last(src);
  if (suggest_weight_block) {
    // suggest block
    return true;
  }

  // suggest plain
  return false;
}

// judge to use block or not for Matmul
static inline bool use_blocked_format_for_matmul(const at::Tensor& src) {
  if (!src.defined() || src.is_sparse()) {
    // suggest plain
    return false;
  }

  if (Settings::I().is_onednn_layout_enabled()) {
    // suggest block
    return true;
  }

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  if (!src_ctx.is_plain()) {
    // suggest block
    return true;
  }

  // suggest plain
  return false;
}

static inline std::vector<int64_t> gen_dummy_input_size_for(
    const at::IntArrayRef weight_sizes,
    const int64_t groups) {
  // weights_dims is 3 for conv1d, 4 for (de)conv2d and 5 for (de)conv3d
  auto input_dim = weight_sizes.size();

  std::vector<int64_t> kernel_size;
  if (5 == input_dim) {
    kernel_size.push_back(weight_sizes[input_dim - 3]);
    kernel_size.push_back(weight_sizes[input_dim - 2]);
  }
  if (4 == input_dim) {
    kernel_size.push_back(weight_sizes[input_dim - 2]);
  }
  kernel_size.push_back(weight_sizes[input_dim - 1]);

  std::vector<int64_t> input_sizes;
  auto ic = weight_sizes[1];

  // batch size is 32
  input_sizes.push_back(32);
  // input channel
  input_sizes.push_back(ic);
  // [important] the factor is 14
  input_sizes.push_back(14 * kernel_size[0]);

  if (4 == input_dim) {
    input_sizes.push_back(14 * kernel_size[1]);
  } else if (5 == input_dim) {
    input_sizes.push_back(14 * kernel_size[1]);
    input_sizes.push_back(14 * kernel_size[2]);
  }

  return input_sizes;
}

void convert_conv_weight_layout(
    const at::Tensor& weight,
    const IntArrayRef padding,
    const IntArrayRef stride,
    IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size);

void convert_convtranspose_weight_layout(
    const at::Tensor& weight,
    const IntArrayRef padding,
    const IntArrayRef stride,
    IntArrayRef dilation,
    const IntArrayRef dst_padding,
    const int64_t groups,
    const IntArrayRef input_size);

at::Tensor convert_linear_weight_layout(
    at::Tensor& weight,
    const IntArrayRef input_size);

} // namespace oneDNN
} // namespace xpu
