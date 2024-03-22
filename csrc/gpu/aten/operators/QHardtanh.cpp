#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <torch/custom_class.h>

#include <quantized/QUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

namespace impl {

template <typename scalar_t, typename underlying_t>
struct qclamp_kernel_functor {
  scalar_t operator()(scalar_t in) const {
    return scalar_t(Numerics<underlying_t>::min(
        Numerics<underlying_t>::max(in.val_, min_q.val_), max_q.val_));
  }

  qclamp_kernel_functor(scalar_t min_q, scalar_t max_q)
      : min_q(min_q), max_q(max_q) {}

 private:
  scalar_t min_q;
  scalar_t max_q;
};

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
    qclamp_kernel_functor<scalar_t, underlying_t> f(min_q, max_q);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  });
}

template <typename scalar_t, typename underlying_t>
struct qclamp_max_kernel_functor {
  scalar_t operator()(scalar_t in) const {
    return scalar_t(Numerics<underlying_t>::min(in.val_, max_q.val_));
  }

  qclamp_max_kernel_functor(scalar_t max_q) : max_q(max_q) {}

 private:
  scalar_t max_q;
};

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
    qclamp_max_kernel_functor<scalar_t, underlying_t> f(max_q);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  });
}

template <typename scalar_t, typename underlying_t>
struct qclamp_min_kernel_functor {
  scalar_t operator()(scalar_t in) const {
    return scalar_t(Numerics<underlying_t>::max(in.val_, min_q.val_));
  }

  qclamp_min_kernel_functor(scalar_t min_q) : min_q(min_q) {}

 private:
  scalar_t min_q;
};

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
    qclamp_min_kernel_functor<scalar_t, underlying_t> f(min_q);
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  });
}

} // namespace impl

Tensor quantized_clamp_impl_dpcpp(
    const Tensor& qx,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  const OptionalDeviceGuard device_guard(device_of(qx));
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

Tensor hardtanh(const Tensor& qx, const Scalar& min, const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl_dpcpp(qx, min, max);
  return qy;
}

Tensor& hardtanh_out(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  result = quantized_clamp_impl_dpcpp(qx, min, max);
  return result;
}

Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl_dpcpp(self, min, max);
  self.copy_(qy);
  return self;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  IPEX_QOP_REGISTER(
      TORCH_SELECTIVE_NAME("quantized::clamp"), quantized_clamp_impl_dpcpp);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
