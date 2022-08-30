#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "comm/Math.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void gcd_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "gcd", [&]() {
    dpcpp_fast_mode_kernel_with_scalars(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return calc_gcd(a, b); });
  });
}

static void lcm_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lcm", [&]() {
    dpcpp_fast_mode_kernel_with_scalars(
        iter, [=](scalar_t a, scalar_t b) -> scalar_t {
          scalar_t g = calc_gcd(a, b);
          return (g == 0) ? 0 : Numerics<scalar_t>::abs(a / g * b);
        });
  });
}

} // namespace impl

at::Tensor gcd(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::gcd_kernel_dpcpp(iter);
  return iter.output();
}

at::Tensor& gcd_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::gcd_kernel_dpcpp(iter);
  return out;
}

at::Tensor& gcd_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::gcd_out(self, other, self);
}

at::Tensor lcm(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::lcm_kernel_dpcpp(iter);
  return iter.output();
}

at::Tensor& lcm_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::lcm_kernel_dpcpp(iter);
  return out;
}

at::Tensor& lcm_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::lcm_out(self, other, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
