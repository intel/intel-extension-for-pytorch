#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <runtime/DPCPP.h>
#include <oneDNN/oneDNN.h>

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static inline dnnl::memory::dims deconv_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_padding,
    int64_t groups) {
  auto dim = input_size.size();
  dnnl::memory::dims output_size(dim);
  auto kernel_size = weight_size.slice(2);

  output_size[0] = input_size[0];
  output_size[1] = weight_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    output_size[d] = (input_size[d] - 1) * stride[d - 2] - 2 * padding[d - 2] +
        (dilation[d - 2] * (kernel_size[d - 2] - 1) + 1) +
        output_padding[d - 2];
  }
  return output_size;
}

static inline dnnl::memory::format_tag deconv_input_fmt(int64_t ndim) {
  return (ndim == 4) ? dnnl::memory::format_tag::nchw
                     : ((ndim == 5) ? dnnl::memory::format_tag::ncdhw
                                    : dnnl::memory::format_tag::undef);
}

static inline dnnl::memory::format_tag deconv_weight_fmt(
    int64_t ndim,
    bool grouped = false) {
  return (ndim == 4)
      ? (grouped ? dnnl::memory::format_tag::giohw
                 : dnnl::memory::format_tag::iohw)
      : ((ndim == 5) ? (grouped ? dnnl::memory::format_tag::giodhw
                                : dnnl::memory::format_tag::iodhw)
                     : dnnl::memory::format_tag::undef);
}

static inline dnnl::memory::dims deconv_compatible_dilation(
    IntArrayRef& dilation) {
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline dnnl::memory::dims deconv_compatible_weight_dims(
    int64_t ndim,
    int64_t groups,
    int64_t oc,
    int64_t ic,
    IntArrayRef wsizes) {
  if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : dnnl::memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1)
        ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : dnnl::memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

Tensor dpcpp_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int64_t groups);

Tensor dpcpp_convolution_transpose_backward_input(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_output,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined);

std::tuple<at::Tensor, at::Tensor> dpcpp_convolution_transpose_backward_weights(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_output,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined);

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
