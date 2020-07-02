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
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> kernel_size{w_dims.cbegin(), w_dims.cend()};
  std::vector<int64_t> output_sizes = calc_deconv_input_size(input_size, kernel_size, padding, output_padding, stride, dilation, groups);

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

}  // namespace deconv
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
