#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include <core/DPCPP.h>


namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

constexpr int input_batch_size_dim = 0;
constexpr int weight_output_channels_dim = 0;

typedef struct conv_attr {
  static const int64_t kind_with_relu = at::dpcpp::oneDNN::with_relu; // 0b01;
  static const int64_t kind_with_sum = at::dpcpp::oneDNN::with_sum; // 0b10;
  static const int64_t kind_with_sigmoid = at::dpcpp::oneDNN::with_sigmoid; // 0b100;

  conv_attr() : scale_(1.f), alpha_(0.f), beta_(0.f), oscale_(1.f), attr_(0) {}
  conv_attr(float scale, float alpha, float beta, float oscale, int64_t attr)
      : scale_(scale), alpha_(alpha), beta_(beta), oscale_(oscale), attr_(attr) {}

  bool with_relu() {
    return attr_ & kind_with_relu;
  }

  bool with_sum() {
    return attr_ & kind_with_sum;
  }

  bool with_sigmoid() {
    return attr_ & kind_with_sigmoid;
  }

  int64_t attr() {
    return attr_;
  }

  float scale_;
  float alpha_;
  float beta_;
  float oscale_;
  int64_t attr_;
} conv_attr_t;

static inline dnnl::memory::dims conv_output_size(
    int64_t ndim,
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  dnnl::memory::dims output_size(ndim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

static inline dnnl::memory::dims compatible_dilation(IntArrayRef &dilation) {
  dnnl::memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline dnnl::memory::format_tag conv_input_fmt(int64_t ndim) {
  return (ndim == 4) ? dnnl::memory::format_tag::nchw
    : ((ndim == 5) ? dnnl::memory::format_tag::ncdhw
        : dnnl::memory::format_tag::undef);
}

static inline dnnl::memory::format_tag conv_weight_fmt(int64_t ndim, bool grouped = false) {
  return (ndim == 4) ? (grouped ? dnnl::memory::format_tag::goihw : dnnl::memory::format_tag::oihw)
    : ((ndim == 5) ? (grouped ? dnnl::memory::format_tag::goidhw : dnnl::memory::format_tag::oidhw)
        : dnnl::memory::format_tag::undef);
}

static inline dnnl::memory::dims compatible_weight_dims(
    int64_t ndim, int64_t groups, int64_t oc, int64_t ic, IntArrayRef wsizes) {
  if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1) ? dnnl::memory::dims({groups, oc / groups, ic / groups, kh, kw})
      : dnnl::memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1) ? dnnl::memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
      : dnnl::memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

at::Tensor convolution(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    conv_attr_t attr);

} // namespace impl
} // namespace AtenIpexTypeDPCPP
} // namespace at
