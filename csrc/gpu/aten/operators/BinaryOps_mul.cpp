#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/zmath.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

static void mul_kernel_dpcpp(TensorIteratorBase& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "mul",
      [&]() {
        fast_mode_opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; });
      });
}
} // namespace impl

Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return binary_out_template<dnnl::algorithm::binary_mul>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) { impl::mul_kernel_dpcpp(iter); });
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  return binary_out_template<dnnl::algorithm::binary_mul>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) { impl::mul_kernel_dpcpp(iter); });
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::mul_out(self, other, self);
}

Tensor mul(const Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::mul(self, wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::mul_out(self, wrapped_scalar_tensor(other), self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
