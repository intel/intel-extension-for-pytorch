#include "Conv.h"
#include "mkldnn/MKLDNNCommon.h"

namespace torch_ipex {
namespace cpu {

namespace {

using weakref_type = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_blocked = std::tuple<weakref_type, ideep::tensor>;
thread_local std::unordered_map<c10::TensorImpl *, val_blocked> cached_weights;

}  // namespace

std::vector<int64_t> calc_conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = kernel_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

ideep::tensor get_prepack_conv_weights(
    const ideep::tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  auto it = cached_weights.find(weight.unsafeGetTensorImpl());
  if (it != cached_weights.end()) {
    return std::get<1>(it->second);
  } else {
    ideep::tensor w = at::native::itensor_view_from_dense(weight);
    // TODO: 3d check
    bool is_channels_last = input.get_desc().is_nhwc();
    ideep::tensor::desc packed_desc;
    if (is_channels_last) {
      packed_desc = ideep::convolution_forward::expected_weights_desc<true>(
        w.get_dims(),
        w.get_data_type(),
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward,
        input.get_data_type(),
        input.get_dims(),
        attr);
    } else {
      packed_desc = ideep::convolution_forward::expected_weights_desc<false>(
        w.get_dims(),
        w.get_data_type(),
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward,
        input.get_data_type(),
        input.get_dims(),
        attr);
    }
    ideep::tensor result;
    result.init(packed_desc);
    result.feed_from(w);
    cached_weights.emplace(
        weight.unsafeGetTensorImpl(),
        val_blocked{weakref_type(weight.getIntrusivePtr()), result});
    return result;
  }
}

at::Tensor convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
// TODO: the input will be actively converted to channels last format
// after the 5-D tensor supports channels last format.
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input);
  ideep::tensor mkldnn_weight = get_prepack_conv_weights(mkldnn_input, weight, stride, padding, dilation, groups, attr);
  auto kernel_size = mkldnn_weight.get_dims();
  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  bool is_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto output = at::empty({0}, input.options());
  ideep::tensor mkldnn_output;
  if (is_channels_last) {
    output.resize_(output_sizes, input.suggest_memory_format());
    mkldnn_output = at::native::itensor_view_from_dense(output);
  }

  if (bias.defined()) {
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias);
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }

  if (is_channels_last) {
    return output;
  } else {
    return at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

void convolution_inplace_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
// TODO: the input will be actively converted to channels last format
// after the 5-D tensor supports channels last format.
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(input);
  ideep::tensor mkldnn_weight = get_prepack_conv_weights(mkldnn_input, weight, stride, padding, dilation, groups, attr);
  auto kernel_size = mkldnn_weight.get_dims();
  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  bool is_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  output.resize_(output_sizes, input.suggest_memory_format());
  ideep::tensor mkldnn_output = at::native::itensor_view_from_dense(output);

  if (bias.defined()) {
    const ideep::tensor mkldnn_bias = at::native::itensor_view_from_dense(bias);
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        {output_sizes.cbegin(), output_sizes.cend()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }

  if (!is_channels_last) {
    output = at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

}  // namespace cpu
}  // namespace torch_ipex
