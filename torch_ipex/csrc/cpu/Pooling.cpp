#include "mkldnn/MKLDNNCommon.h"
#include "torch_ipex/csrc/utils.h"

#include <ATen/native/Pool.h>
#include <ATen/core/grad_mode.h>

namespace torch_ipex {
namespace cpu {

inline std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

std::vector<int64_t> pool_output_sizes(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding_l,
    at::IntArrayRef padding_r,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());
  // copy N and C
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  for (size_t i = 2; i < input_size.size(); ++i) {
    output_size[i] = at::native::pooling_output_shape_pad_lr<int64_t>(
      input_size[i],
      kernel_size[i - 2],
      padding_l[i - 2],
      padding_r[i - 2],
      stride[i - 2],
      dilation[i - 2],
      ceil_mode
    );
  }

   return output_size;
}

at::Tensor pooling_impl(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    ideep::algorithm algo) {

  const int64_t dims = input.dim() - 2;
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  if (stride.empty()) stride = kernel_size;
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

// TODO: the input will be actively converted to channels last format
// after the 5-D tensor supports channels last format.
  auto input_ = IS_CONTIGUOUS_ANY(input) ? input : input.contiguous();
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input_);
  std::vector<int64_t> output_sizes;

  if (ceil_mode) {
    // MKLDNN does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // adjust padding until output sizes agree
    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /*ceil_mode */);

      all_equal = true;
      for (size_t i = 2; i < input.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  } else {
    output_sizes = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        false /*ceil_mode */);
  }

  bool is_channels_last = input_.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto output = at::empty(output_sizes, input_.options().memory_format(input_.suggest_memory_format()));
  ideep::tensor mkldnn_output;
  if (is_channels_last) {
    mkldnn_output = at::native::itensor_view_from_dense(output);
  }

  auto aprop_kind = ideep::prop_kind::forward;
  // for max_pool, prop_kind::forward will save indices as workspace for backward use,
  // for inference, don't need the indices, set aprop_kind to prop_kind::forward_inference
  // can reduce the memory use.
  if (ideep::algorithm::pooling_max == algo
      && !(input.requires_grad() && at::GradMode::is_enabled())) {
    aprop_kind = ideep::prop_kind::forward_inference;
  }

  ideep::tensor y;
  ideep::pooling_forward::compute(
      mkldnn_input,
      {output_sizes.cbegin(), output_sizes.cend()},
      mkldnn_output,
      {stride_vec.cbegin(), stride_vec.cend()},
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algo,
      aprop_kind);

  if (is_channels_last) {
    return output;
  } else {
    return at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

}  // namespace cpu
}  // namespace torch_ipex
