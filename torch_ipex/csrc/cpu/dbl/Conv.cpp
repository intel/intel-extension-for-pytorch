#include "Conv.h"

#include "Common.h"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace conv {

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

dil::tensor convolution_impl(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr) {
  std::vector<int64_t> kernel_size(x.ndims());
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(
      groups > 1,
      "Only group _mkldnn_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(w.get_dims().cbegin() + 2, x.ndims() - 1, kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());
  }

  const dil::dims x_dims = x.get_dims();
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> output_sizes = calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  dil::tensor y;
  if (b.has_value()) {
    dil::convolution_forward::compute(
      x,
      w,
      b.value(),
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      dil::scale_t(),
      dil::scale_t(),
      dil::scale_t(),
      attr);
  } else {
    dil::convolution_forward::compute(
      x,
      w,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      dil::scale_t(),
      dil::scale_t(),
      dil::scale_t(),
      attr);
  }
  return y;
}

void convolution_inplace_impl(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    dil::tensor& y,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr) {
  std::vector<int64_t> kernel_size(x.ndims());
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(
      groups > 1,
      "Only group _mkldnn_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(w.get_dims().cbegin() + 2, x.ndims() - 1, kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());
  }

  const dil::dims x_dims = x.get_dims();
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> output_sizes = calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  if (b.has_value()) {
    dil::convolution_forward::compute(
      x,
      w,
      b.value(),
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      dil::scale_t(),
      dil::scale_t(),
      dil::scale_t(),
      attr);
  } else {
    dil::convolution_forward::compute(
      x,
      w,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      dil::scale_t(),
      dil::scale_t(),
      dil::scale_t(),
      attr);
  }
}

void prepack_conv_weights(
    const at::Tensor& input,
    const dil::tensor& dil_input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  // Prepack weight tensor if it's either a *cpu tensor* or a *plain dil tensor*
  //
  // Note: weight tensor will not be re-packed unless user has implicitly
  //       triggered `to_public` by accessing its data
  //       One caveat is when the input size has changed and prepacked weight
  //       might not be the best fit for new input size, the weight will not
  //       be re-packed in such cases, but it still ensures the correctness
  //
  // TODO: once semantics of "own shade context" is equivalent to
  //       "is dil tensor", we could remove the first check below
  if (!check_tensor_own_shade_context(weight) ||
      !cpu::ShadeDataContext::isDilOwnTheTensor(weight) ||
      cpu::ShadeDataContext::getDilTensor(weight).is_public_format()) {
    auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);
    auto packed_desc = dil::convolution_forward::expected_weights_desc(
      weight.sizes().vec(),
      dil_weight.get_data_type(),
      stride.vec(),
      padding.vec(),
      padding.vec(),
      dilation.vec(),
      groups,
      dil::algorithm::convolution_direct,
      dil::prop_kind::forward,
      dil_input.get_data_type(),
      input.sizes().vec());

    if (packed_desc.is_default()) {
      // In some cases of grouped conv, there's no optimized kernel using 
      // blocked weight. So if queried format is still a public (plain) format,
      // we should skip the plain-to-plain reorder. (e.g. g8ic32oc80sp7k3)
      // 
      // TODO: Now we're deciding whether to prepack weight based on if it's
      // already been a blocked tensor. But if its optimzal format is not
      // blocked, we are wasting time to query format on each call because of
      // pd-creation overhead.
      return;
    }

    dil::tensor packed_weight {packed_desc};
    packed_weight.feed_from(dil_weight);
    dbl::comm::equip_dil_buffer(weight, packed_weight);
  }
}

}  // namespace conv
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
