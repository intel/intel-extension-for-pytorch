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

Scalar _local_scalar_dense(const Tensor& self);
Tensor min(const Tensor& self);

namespace impl {

template <typename scalar_t>
struct EqKernelDpcppFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return Numerics<scalar_t>::eq(a, b);
  }
};

void eq_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      iter.common_dtype(),
      "eq_dpcpp",
      [&]() {
        EqKernelDpcppFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
}

} // namespace impl

/*=========================== eq ==========================*/

Tensor& eq_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::eq_kernel_dpcpp(iter);
  return out;
}

bool equal(const Tensor& self, const Tensor& other) {
  if (!self.sizes().equals(other.sizes()))
    return false;

  if (self.numel() == 0)
    return true;

  Tensor result = at::empty_like(self, self.options().dtype(kBool));

  at::AtenIpexTypeXPU::eq_out(self, other, result);
  Tensor min = at::AtenIpexTypeXPU::min(result);
  Scalar min_ = at::AtenIpexTypeXPU::_local_scalar_dense(min);
  return min_.to<bool>() != 0;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
bool equal(const Tensor& self, const Tensor& other) {
  if (!other.is_quantized()) {
    return false;
  }

  // Delegate to virtual equalTo method. This will ensure different concrete
  // Quantizers can have specific logic for comparison
  auto self_quantizer = get_qtensorimpl(self)->quantizer();
  auto other_quantizer = get_qtensorimpl(other)->quantizer();
  if (!self_quantizer->equalTo(other_quantizer)) {
    return false;
  }

  // Sizes and element types must be the same
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.element_size() != other.element_size()) {
    return false;
  }

  if (self.numel() == 0)
    return true;

  return at::AtenIpexTypeXPU::equal(self, other);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
