#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>
#include <ATen/quantized/QTensorImpl.h>

#include <ATen/native/TensorIterator.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct NeKernelDpcppFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::ne(a, b);
  }
};

void ne_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "ne_dpcpp",
      [&]() {
        NeKernelDpcppFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
}

} // namespace impl

/*=========================== ne ==========================*/

Tensor ne(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  auto iter = TensorIterator::comparison_op(result, self, other);
  impl::ne_kernel_dpcpp(iter);
  return result;
}

Tensor ne(const Tensor& self, const Scalar& other_) {
  return at::AtenIpexTypeXPU::ne(self, wrapped_scalar_tensor(other_));
}

} // namespace AtenIpexTypeXPU
} // namespace at
