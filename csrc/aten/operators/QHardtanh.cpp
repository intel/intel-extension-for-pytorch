#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <torch/custom_class.h>

#include <quantized/QUtil.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

namespace impl {

void qclamp_kernel(
    const Tensor& qx,
    const Scalar& min_scalar,
    const Scalar& max_scalar,
    Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&] {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kXPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);

    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    auto max = max_scalar.to<float>();
    scalar_t min_q =
        quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), min);
    scalar_t max_q =
        quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), max);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t in) -> scalar_t {
          return scalar_t(Numerics<underlying_t>::min(
              Numerics<underlying_t>::max(in.val_, min_q.val_), max_q.val_));
        });
  });
}

void qclamp_max_kernel(const Tensor& qx, const Scalar& max_scalar, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp_max", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kXPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);
    auto max = max_scalar.to<float>();
    scalar_t max_q =
        quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), max);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t in) -> scalar_t {
          return scalar_t(Numerics<underlying_t>::min(in.val_, max_q.val_));
        });
  });
}

void qclamp_min_kernel(const Tensor& qx, const Scalar& min_scalar, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp_min", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kXPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    scalar_t min_q =
        quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(), min);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t in) -> scalar_t {
          return scalar_t(Numerics<underlying_t>::max(in.val_, min_q.val_));
        });
  });
}

} // namespace impl

Tensor quantized_clamp_impl_dpcpp(
    const Tensor& qx,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  Tensor qy;
  if (min && max) {
    impl::qclamp_kernel(qx, *min, *max, qy);
  } else {
    if (max) {
      impl::qclamp_max_kernel(qx, *max, qy);
    } else if (min) {
      impl::qclamp_min_kernel(qx, *min, qy);
    } else {
      TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
    }
  }
  return qy;
}

Tensor hardtanh_quantized_xpu(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl_dpcpp(qx, min, max);
  return qy;
}

Tensor& hardtanh_out_quantized_xpu(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  result = quantized_clamp_impl_dpcpp(qx, min, max);
  return result;
}

Tensor& hardtanh_quantized_xpu_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl_dpcpp(self, min, max);
  self.copy_(qy);
  return self;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::clamp"),
      TORCH_FN(quantized_clamp_impl_dpcpp));
}

TORCH_LIBRARY_IMPL(aten, QuantizedXPU, m) {
  m.impl("hardtanh", hardtanh_quantized_xpu);
  m.impl("hardtanh.out", hardtanh_out_quantized_xpu);
  m.impl("hardtanh_", hardtanh_quantized_xpu_);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
