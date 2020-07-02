#include "Deconv.h"

#include "Common.h"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace deconv {

std::vector<int64_t> calc_deconv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[0];
  input_size[1] = kernel_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

dil::tensor deconvolution_impl(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr) {
  const dil::dims x_dims = x.get_dims();
  const dil::dims w_dims = w.get_dims();
  std::vector<int64_t> output_sizes = calc_deconv_input_size(x_dims, w_dims, padding, output_padding, stride, dilation, groups);

  dil::tensor y;
  if (b.has_value()) {
    dil::convolution_transpose_forward::compute(
      x,
      w,
      b.value(),
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      {dilation.begin(), dilation.end()}, 
      groups,
      attr);
  } else {
    dil::convolution_transpose_forward::compute(
      x,
      w,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      {dilation.begin(), dilation.end()}, 
      groups,
      attr);
  }
  return y;
}

void prepack_deconv_weights(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    int64_t groups,
    bool with_bias) {
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
    auto output_sizes = calc_deconv_input_size(input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
    auto packed_desc = dil::convolution_transpose_forward::expected_weights_desc(
        weight.sizes().vec(),
        dil_weight.get_data_type(),
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups,
        dil::algorithm::deconvolution_direct,
        dil::prop_kind::forward,
        input.sizes().vec(),
        output_sizes,
        with_bias);

    if (packed_desc.is_default()) {
      // In some cases of grouped deconv, there's no optimized kernel using
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
    packed_weight.feed_from(dil_weight, /*is_deconv_weights=*/true);
    dbl::comm::equip_dil_buffer(weight, packed_weight);
  }
}

}  // namespace deconv
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
