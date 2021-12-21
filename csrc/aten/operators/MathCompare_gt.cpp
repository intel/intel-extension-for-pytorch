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

void gt_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "gt_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::gt(a, b);
        });
      });
}

} // namespace impl

/*=========================== gt ==========================*/

Tensor& gt_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::gt_kernel_dpcpp(iter);
  return out;
}

Tensor gt(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::gt_out(result, self, other);
}

Tensor& gt_out(Tensor& out, const Tensor& self, Scalar other_) {
  at::AtenIpexTypeXPU::gt_out(out, self, wrapped_scalar_tensor(other_));
  return out;
}

Tensor gt(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::gt_out(
      result, self, wrapped_scalar_tensor(other_));
}

} // namespace AtenIpexTypeXPU
} // namespace at
