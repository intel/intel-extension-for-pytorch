#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <core/DPCPP.h>

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

constexpr int input_batch_size_dim = 0;
constexpr int weight_output_channels_dim = 0;

typedef struct conv_attr {
  static const int64_t kind_with_relu = 0b01;
  static const int64_t kind_with_sum = 0b10;

  conv_attr() : scale_(1.0), alpha_(0.f), beta_(0.f), attr_(0) {}
  conv_attr(float scale, float alpha, float beta, int64_t attr)
      : scale_(scale), alpha_(alpha), beta_(beta), attr_(attr) {}

  bool with_relu() {
    return attr_ & kind_with_relu;
  }

  bool with_sum() {
    return attr_ & kind_with_sum;
  }

  float scale_;
  float alpha_;
  float beta_;
  int64_t attr_;
} conv_attr_t;

static std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
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
