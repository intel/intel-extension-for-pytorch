#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

#include <ATen/Context.h>
#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include <utils/CustomOperatorRegistration.h>
#endif
#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct special_ndtri_out_functor {
  scalar_t operator()(scalar_t a) const {
    return calc_ndtri(a);
  }
};

template <typename scalar_t>
struct special_erfcx_out_functor {
  scalar_t operator()(scalar_t a) const {
    return calc_erfcx(a);
  }
};

#ifdef USE_OVERRIDE_OP
at::Tensor special_erfcx(const at::Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "erfcx", [&]() {
        special_erfcx_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return iter.output();
}

at::Tensor special_ndtri(const at::Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "special_ndtri", [&]() {
        special_ndtri_out_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return iter.output();
}
#endif
} // namespace AtenIpexTypeXPU
} // namespace at
#ifdef USE_OVERRIDE_OP
at::Tensor wrapper_XPU_special_erfcx(const at::Tensor& self) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_special_erfcx", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::special_erfcx(self);
}

at::Tensor wrapper_XPU_special_ndtri(const at::Tensor& self) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_special_ndtri", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::special_ndtri(self);
}

namespace {
IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("special_erfcx", TORCH_FN((&wrapper_XPU_special_erfcx)));
  m.impl("special_ndtri", TORCH_FN((&wrapper_XPU_special_ndtri)));
}
} // namespace
#endif
