#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>
#include <ATen/quantized/QTensorImpl.h>

#include <ATen/native/TensorIterator.h>
#include <core/TensorImplUtils.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/ScalarOps.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void eq_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "eq_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::eq(a, b);
        });
      });
}

} // namespace impl

/*=========================== eq ==========================*/

Tensor& eq_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::eq_kernel_dpcpp(iter);
  return out;
}

Tensor eq(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::eq_out(result, self, other);
}
Tensor& eq_out(Tensor& out, const Tensor& self, Scalar other_) {
  return at::AtenIpexTypeXPU::eq_out(out, self, wrapped_scalar_tensor(other_));
}

Tensor eq(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::eq_out(
      result, self, wrapped_scalar_tensor(other_));
}

bool equal(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty_like(self, self.options().dtype(kBool));

  if (!self.sizes().equals(other.sizes()))
    return false;

  at::AtenIpexTypeXPU::eq_out(result, self, other);
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

  return at::AtenIpexTypeXPU::equal(self, other);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
