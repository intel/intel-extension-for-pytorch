#include <ATen/ATen.h>
#include <oneDNN/oneDNN.h>
#include <torch/library.h>

using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

static inline Tensor repeat_if_defined_IN(const Tensor& t, int64_t repeat) {
  if (t.defined()) {
    return t.repeat(repeat);
  }
  return t;
}

at::Tensor quantized_native_instance_norm(
    const at::Tensor& qinput,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
  auto input = at::dequantize(qinput);

  std::vector<int64_t> shape = input.sizes().vec();
  int64_t b = input.size(0);
  int64_t c = input.size(1);
  shape[1] = b * c;
  shape[0] = 1;

  Tensor weight_ = repeat_if_defined_IN(weight, b);
  Tensor bias_ = repeat_if_defined_IN(bias, b);
  Tensor running_mean_;
  Tensor running_var_;

  auto input_reshaped = input.contiguous().view(shape);
  auto output = at::batch_norm(
      input_reshaped,
      weight_,
      bias_,
      running_mean_,
      running_var_,
      true,
      0,
      eps,
      false);

  auto q_out = at::quantize_per_tensor(
      output, output_scale, output_zero_point, qinput.scalar_type());

  return q_out.view(qinput.sizes());
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::instance_norm"),
      [](Tensor qx,
         c10::optional<Tensor> weight,
         c10::optional<Tensor> bias,
         double eps,
         double output_scale,
         int64_t output_zero_point) {
        return quantized_native_instance_norm(
            qx,
            weight.has_value() ? *weight : Tensor(),
            bias.has_value() ? *bias : Tensor(),
            eps,
            output_scale,
            output_zero_point);
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
