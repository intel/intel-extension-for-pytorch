#include <ATen/ATen.h>

#include <oneDNN/oneDNN.h>

using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

static inline at::Tensor condition_contiguous(const at::Tensor& t) {
  if (t.defined() && !t.is_contiguous(at::MemoryFormat::ChannelsLast))
    return t.contiguous();
  return t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double epsilon) {
  checkBackend(
      "batch_norm",
      {input, weight, bias, running_mean, running_var},
      Backend::XPU);

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  } else {
    return xpu::oneDNN::batch_normalization(
        condition_contiguous(input),
        condition_contiguous(weight),
        condition_contiguous(bias),
        condition_contiguous(running_mean),
        condition_contiguous(running_var),
        training,
        momentum,
        epsilon);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool training,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  checkBackend(
      "batch_norm",
      {input, weight, grad_output, running_mean, running_var, save_mean, save_var},
      Backend::XPU);

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  } else {
    return xpu::oneDNN::batch_normalization_backward(
        condition_contiguous(grad_output),
        condition_contiguous(input),
        condition_contiguous(weight),
        condition_contiguous(running_mean),
        condition_contiguous(running_var),
        condition_contiguous(save_mean),
        condition_contiguous(save_var),
        training,
        epsilon,
        grad_input_mask);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
