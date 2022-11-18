#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <oneapi/dpl/tuple>
#include "Loops.h"

using namespace xpu::dpcpp;

/* Fake quantize a tensor
Args:
  output: output tensor.
  input : input tensor.
  sc:  scale to quantize the input tensor to
  zero_point: zero_point
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (float dtype).
*/

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

int64_t _get_zero_point_from_tensor(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    bool is_forward) {
  float zero_point_fp = zero_point[0].item<float>();
  zero_point_fp =
      is_forward ? std::nearbyint(zero_point_fp) : zero_point_fp + 0.5f;
  float zero_point_clamped = std::min(
      std::max(zero_point_fp, static_cast<float>(quant_min)),
      static_cast<float>(quant_max));
  return static_cast<int64_t>(zero_point_clamped);
}

Tensor fake_quantize_per_tensor_affine(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  const auto res = at::fake_quantize_per_tensor_affine_cachemask(
      self, scale, zero_point, quant_min, quant_max);
  return std::get<0>(res);
}

Tensor fake_quantize_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  const auto res =
      at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
          self,
          scale,
          zero_point,
          at::ones(1, self.options().dtype(at::kLong)),
          quant_min,
          quant_max);
  return std::get<0>(res);
}

void fake_quantize_tensor_cachemask_dpcpp(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float inv_scale = 1.0f / scale;
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_output(mask)
                  .add_input(input)
                  .build();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fake_quantize_tensor_cachemask_dpcpp",
      [&] {
        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter, [=](scalar_t input_val) -> dpl::tuple<scalar_t, bool> {
              const auto qval = static_cast<int64_t>(
                  dpl::nearbyint(input_val * inv_scale) + zero_point);
              return {// fake_quantized value
                      (Numerics<int64_t>::min(
                           quant_max, Numerics<int64_t>::max(quant_min, qval)) -
                       zero_point) *
                          scale,
                      // mask for grad
                      ((quant_min <= qval) && (qval <= quant_max))};
            });
      });
}

void _fake_quantize_grad_learnable_tensor_dpcpp(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  float dscale_small = quant_min - zero_point;
  float dscale_big = quant_max - zero_point;
  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](float XInput, float dYInput) -> dpl::tuple<float, float, float> {
        float dXOutput, dZeroPointOutput, dScaleOutput;
        int64_t xq = dpl::nearbyint(XInput * inv_scale) + zero_point;
        dXOutput = dYInput * (xq >= quant_min && xq <= quant_max);
        float xfq = static_cast<float>(
            (Numerics<int64_t>::max(
                 Numerics<int64_t>::min(xq, quant_max), quant_min) -
             zero_point) *
            scale);
        if (xq < quant_min || xq > quant_max) {
          dZeroPointOutput = (dYInput) * (-1) * scale * grad_factor;
          dScaleOutput = ((xq < quant_min) ? (dYInput * dscale_small)
                                           : (dYInput * dscale_big)) *
              grad_factor;
        } else {
          dZeroPointOutput = 0;
          dScaleOutput = (dYInput) * (xfq - (XInput)) * inv_scale * grad_factor;
        }
        return {dXOutput, dScaleOutput, dZeroPointOutput};
      });
}

void _fake_quantize_tensor_cachemask_tensor_qparams_dpcpp(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();
  int64_t* fake_quant_on = fake_quant_enabled.data_ptr<int64_t>();
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_output(mask)
                  .add_input(input)
                  .build();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fake_quantize_tensor_cachemask_tensor_qparams_dpcpp",
      [&] {
        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter, [=](scalar_t input_val) -> dpl::tuple<scalar_t, bool> {
              if (*fake_quant_on == 0) {
                return {input_val, 1};
              }
              float inv_scale = 1.0f / (*scale_ptr);
              const auto qval = static_cast<int64_t>(
                  dpl::nearbyint(input_val * inv_scale) + (*zp_ptr));
              return {// fake_quantized value
                      (Numerics<int64_t>::min(
                           quant_max, Numerics<int64_t>::max(quant_min, qval)) -
                       (*zp_ptr)) *
                          (*scale_ptr),
                      // mask for grad
                      ((quant_min <= qval) && (qval <= quant_max))};
            });
      });
}

} // namespace impl

Tensor _fake_quantize_learnable_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  float scale_val = scale[0].item<float>();
  int64_t zero_point_val =
      impl::_get_zero_point_from_tensor(zero_point, quant_min, quant_max, true);
  return impl::fake_quantize_per_tensor_affine(
      self, scale_val, zero_point_val, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  /* The gradients for scale and zero point are calculated as below:
     Let Xfq be the fake quantized version of X.
     Let Xq be the quantized version of X (clamped at qmin and qmax).
     Let Delta and z be the scale and the zero point.
     :math:
      \frac{d\Delta }{dx} =
        \begin{cases}
          q_{\min} - z& \text{ if } X_q= q_{\min} \\
          q_{\max} - z& \text{ if } X_q= q_{\max} \\
          (X_{fq} - X) / \Delta & \text{ else }
        \end{cases}

      \frac{dz }{dx} =
        \begin{cases}
          -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
          0 & \text{ else }
        \end{cases}
  */
  float scale_val = scale[0].item<float>();
  float inv_scale_val = 1.0f / scale_val;
  int64_t zero_point_val = impl::_get_zero_point_from_tensor(
      zero_point, quant_min, quant_max, false);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "`quant_min` should be less than or \
        equal to `quant_max`, and the quantization range should include 0.");
  TORCH_CHECK(
      zero_point_val >= quant_min && zero_point_val <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);

  auto iter = TensorIteratorConfig()
                  .add_output(dX)
                  .add_output(dScale_vec)
                  .add_output(dZeroPoint_vec)
                  .add_input(X)
                  .add_input(dY)
                  .build();

  impl::_fake_quantize_grad_learnable_tensor_dpcpp(
      iter,
      scale_val,
      inv_scale_val,
      zero_point_val,
      quant_min,
      quant_max,
      grad_factor);

  // The total sums over the scale and zero point gradient vectors are what will
  // be returned in the end.
  auto dScale = dScale_vec.sum().unsqueeze(0).to(scale.device());
  auto dZeroPoint = dZeroPoint_vec.sum().unsqueeze(0).to(zero_point.device());

  return std::make_tuple(dX, dScale, dZeroPoint);
}

/* Fake-quantizes the 'inputs' tensor, saving a mask for the backward pass.

This is numerically equivalent to `fake_quantize_per_tensor_affine`,
but has a lower memory overhead in the backward pass.

Args:
  self: Forward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value

Returns:
  Quantized tensor (double dtype).
  Mask (bool dtype).
*/
std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  impl::fake_quantize_tensor_cachemask_dpcpp(
      Y, mask, self, scale, zero_point, quant_min, quant_max);
  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  return std::make_tuple(Y, mask);
}

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  impl::_fake_quantize_tensor_cachemask_tensor_qparams_dpcpp(
      Y,
      mask,
      self,
      scale,
      zero_point,
      fake_quant_enabled,
      quant_min,
      quant_max);
  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  return std::make_tuple(Y, mask);
}

} // namespace AtenIpexTypeXPU
} // namespace at
