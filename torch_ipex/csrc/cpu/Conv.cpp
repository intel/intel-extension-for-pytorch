#include "Conv.h"
#include "mkldnn/MKLDNNCommon.h"

namespace torch_ipex {
namespace cpu {

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

at::Tensor convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {

  const ideep::tensor mkldnn_input = at::native::itensor_from_tensor(input);
  const ideep::tensor mkldnn_weight = at::native::itensor_from_tensor(weight);

  auto kernel_size = mkldnn_weight.get_dims();

  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  ideep::tensor mkldnn_output;
  if (bias.defined()) {
    const ideep::tensor mkldnn_bias = at::native::itensor_from_tensor(bias);
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

  return at::native::mkldnn_to_dense(
      at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()));
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

  const ideep::tensor mkldnn_input = at::native::itensor_from_tensor(input);
  const ideep::tensor mkldnn_weight = at::native::itensor_from_tensor(weight);

  auto kernel_size = mkldnn_weight.get_dims();

  std::vector<int64_t> input_size = mkldnn_input.get_dims();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  ideep::tensor mkldnn_output = at::native::itensor_from_tensor(output);
  if (bias.defined()) {
    const ideep::tensor mkldnn_bias = at::native::itensor_from_tensor(bias);
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
  output = at::native::mkldnn_to_dense(
      at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()));
}

// void prepack_conv_weights(
//     const at::Tensor& input,
//     const at::Tensor& dil_input,
//     const at::Tensor& weight,
//     at::IntArrayRef stride,
//     at::IntArrayRef padding,
//     at::IntArrayRef dilation,
//     int64_t groups) {
//   // Prepack weight tensor if it's either a *cpu tensor* or a *plain dil tensor*
//   //
//   // Note: weight tensor will not be re-packed unless user has implicitly
//   //       triggered `to_public` by accessing its data
//   //       One caveat is when the input size has changed and prepacked weight
//   //       might not be the best fit for new input size, the weight will not
//   //       be re-packed in such cases, but it still ensures the correctness
//   //
//   // TODO: once semantics of "own shade context" is equivalent to
//   //       "is dil tensor", we could remove the first check below
//   if (!cpu::ShadeDataContext::isPackedTensor(weight)) {
//     auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);
//     auto packed_desc = ideep::convolution_forward::expected_weights_desc(
//       weight.sizes().vec(),
//       dil_weight.get_data_type(),
//       stride.vec(),
//       padding.vec(),
//       padding.vec(),
//       dilation.vec(),
//       groups,
//       ideep::algorithm::convolution_direct,
//       ideep::prop_kind::forward,
//       dil_input.get_data_type(),
//       input.sizes().vec());

//     at::Tensor packed_weight {packed_desc};
    
//     if (dil_weight.has_scale()) {
//       packed_weight.set_scale(dil_weight.get_scale());
//     }
//     packed_weight.feed_from(dil_weight);
//     dbl::comm::equip_dil_buffer(weight, packed_weight);
//     cpu::ShadeDataContext::setPackedTensor(weight, true);
//   }
// }

}  // namespace cpu
}  // namespace torch_ipex
