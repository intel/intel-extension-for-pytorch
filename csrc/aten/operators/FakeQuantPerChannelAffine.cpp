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

Tensor _get_rounded_zero_point(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // This assumes the per channel zero point vector is single-dimensioned.
  return zero_point.round().clamp_(quant_min, quant_max);
}

void fake_quantize_per_channel_cachemask_dpcpp(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  // for simplicity, as we do not expect this to be a bottleneck.
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "fake_quantize_per_channel_cachemask_dpcpp",
      [&] {
        // write mask
        dpcpp_kernel_for_tensor_iter(
            iter_mask,
            [=](scalar_t input_val, float scale, int64_t zero_point) -> bool {
              float inv_scale = 1.0f / scale;
              const auto qval = static_cast<int64_t>(
                  dpl::nearbyint(input_val * inv_scale) + zero_point);
              return ((quant_min <= qval) && (qval <= quant_max));
            });

        // write fake_quant
        dpcpp_kernel_for_tensor_iter(
            iter,
            [=](scalar_t input_val,
                float scale,
                int64_t zero_point) -> scalar_t {
              float inv_scale = 1.0f / scale;
              return (Numerics<int64_t>::min(
                          quant_max,
                          Numerics<int64_t>::max(
                              quant_min,
                              static_cast<int64_t>(
                                  dpl::nearbyint(input_val * inv_scale) +
                                  zero_point))) -
                      zero_point) *
                  scale;
            });
      });
}

Tensor fake_quantize_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  const auto res = at::fake_quantize_per_channel_affine_cachemask(
      self, scale, zero_point, axis, quant_min, quant_max);
  return std::get<0>(res);
}

void _fake_quantize_grad_learnable_channel_dpcpp(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](float x_input,
          float dy_input,
          float scale_input,
          float zero_point_input) -> dpl::tuple<float, float, float> {
        float dx_output, dscale_output, dzero_point_output;
        float inv_scale = 1.0f / scale_input;
        float dscale_small = quant_min - zero_point_input;
        float dscale_big = quant_max - zero_point_input;
        // Calculate gradients for X.
        int64_t xqi = dpl::nearbyint(x_input * inv_scale) +
            static_cast<int64_t>(zero_point_input);
        dx_output = dy_input * (xqi >= quant_min && xqi <= quant_max);
        // Calculate gradients for scale and zero point.
        float xfqi = static_cast<float>(
            (Numerics<int64_t>::max(
                 Numerics<int64_t>::min(xqi, quant_max), quant_min) -
             zero_point_input) *
            scale_input);
        if (xqi < quant_min || xqi > quant_max) {
          dzero_point_output = dy_input * (-1) * scale_input * grad_factor;
          dscale_output = ((xqi < quant_min) ? (dy_input * dscale_small)
                                             : (dy_input * dscale_big)) *
              grad_factor;
        } else {
          dzero_point_output = 0;
          dscale_output = dy_input * (xfqi - x_input) * inv_scale * grad_factor;
        }
        return {dx_output, dscale_output, dzero_point_output};
      });
}

} // namespace impl

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(
      zero_point.scalar_type() == ScalarType::Int ||
          zero_point.scalar_type() == ScalarType::Float ||
          zero_point.scalar_type() == ScalarType::Half,
      "Zero-point must be Int32, Float or Half, found ",
      zero_point.scalar_type());
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == self.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

  TORCH_CHECK(
      at::min(zero_point).item().toInt() >= quant_min &&
          at::max(zero_point).item().toInt() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis <= self.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);

  std::vector<int64_t> expected_shape(self.dim(), 1);
  expected_shape[axis] = self.size(axis);

  TensorIterator iter = TensorIteratorConfig()
                            .check_all_same_dtype(false)
                            .add_output(Y)
                            .add_input(self)
                            .add_owned_input(at::AtenIpexTypeXPU::_unsafe_view(
                                scale, expected_shape))
                            .add_owned_input(at::AtenIpexTypeXPU::_unsafe_view(
                                zero_point, expected_shape))
                            .build();

  // TODO(future, optional): read once, write twice.  Not done at the moment
  // for simplicity, as we do not expect this to be a bottleneck.
  TensorIterator iter_mask =
      TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(mask)
          .add_input(self)
          .add_owned_input(
              at::AtenIpexTypeXPU::_unsafe_view(scale, expected_shape))
          .add_owned_input(
              at::AtenIpexTypeXPU::_unsafe_view(zero_point, expected_shape))
          .build();

  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  // 1 byte per element, we only need 1 bit per element).
  impl::fake_quantize_per_channel_cachemask_dpcpp(
      iter, iter_mask, quant_min, quant_max);
  return std::make_tuple(Y, mask);
}

Tensor _fake_quantize_learnable_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  Tensor zero_point_rounded =
      impl::_get_rounded_zero_point(zero_point, quant_min, quant_max)
          .to(at::kInt);
  return impl::fake_quantize_per_channel_affine(
      self, scale, zero_point_rounded, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
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
  auto zero_point_rounded =
      impl::_get_rounded_zero_point(zero_point, quant_min, quant_max);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X.sizes() == dY.sizes(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "Expecting `quant_min` <= 0 and `quant_max` >= 0");
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == X.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      at::min(zero_point_rounded).item().toLong() >= quant_min &&
          at::max(zero_point_rounded).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis < X.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  int numDimensions = X.ndimension();

  // Create an axis mask for vectorizing and reshaping the scale and zero point
  // tensors into the same shapes as X along the channel axis.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  int64_t* axis_mask = (int64_t*)calloc(numDimensions, sizeof(int64_t));
  for (int i = 0; i < numDimensions; ++i) {
    axis_mask[i] = (i == axis) ? X.size(axis) : 1;
  }
  auto X_shape = X.sizes();
  auto scale_vectorized =
      scale.reshape(at::IntArrayRef(axis_mask, numDimensions)).expand(X_shape);
  auto zero_point_vectorized =
      zero_point_rounded.reshape(at::IntArrayRef(axis_mask, numDimensions))
          .expand(X_shape);

  auto iter = TensorIteratorConfig()
                  .add_output(dX)
                  .add_output(dScale_vec)
                  .add_output(dZeroPoint_vec)
                  .add_input(X)
                  .add_input(dY)
                  .add_input(scale_vectorized)
                  .add_input(zero_point_vectorized)
                  .build();

  impl::_fake_quantize_grad_learnable_channel_dpcpp(
      iter, quant_min, quant_max, grad_factor);

  auto numElements = X.ndimension() - 1;

  // Create a collection of axes that include all but the channel axis for
  // reduction when summing over the dScale and dZeroPoint tensors.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  int64_t* axis_for_reduction = (int64_t*)calloc(numElements, sizeof(int64_t));
  for (const auto i : c10::irange(axis)) {
    axis_for_reduction[i] = i;
  }
  for (const auto i : c10::irange(axis, numElements)) {
    axis_for_reduction[i] = i + 1;
  }

  auto dScale =
      dScale_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));
  auto dZeroPoint =
      dZeroPoint_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));

  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(axis_mask);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(axis_for_reduction);
  return std::make_tuple(dX, dScale, dZeroPoint);
}

} // namespace AtenIpexTypeXPU
} // namespace at
