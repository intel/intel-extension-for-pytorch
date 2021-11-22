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
void lt_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "lt_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::lt(a, b);
        });
      });
}

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

void ge_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "ge_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::ge(a, b);
        });
      });
}

void le_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "le_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::le(a, b);
        });
      });
}

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

void ne_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "ne_dpcpp",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [=](scalar_t a, scalar_t b) -> bool {
          return Numerics<scalar_t>::ne(a, b);
        });
      });
}

} // namespace impl

/*=========================== lt ==========================*/

Tensor& lt_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::lt_kernel_dpcpp(iter);
  return out;
}

Tensor lt(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::lt_out(result, self, other);
}

Tensor& lt_out(Tensor& out, const Tensor& self, Scalar other_) {
  at::AtenIpexTypeXPU::lt_out(out, self, wrapped_scalar_tensor(other_));
  return out;
}

Tensor lt(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::lt_out(
      result, self, wrapped_scalar_tensor(other_));
}

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

/*=========================== ge ==========================*/

Tensor& ge_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::ge_kernel_dpcpp(iter);
  return out;
}

Tensor ge(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ge_out(result, self, other);
}

Tensor& ge_out(Tensor& out, const Tensor& self, Scalar other_) {
  at::AtenIpexTypeXPU::ge_out(out, self, wrapped_scalar_tensor(other_));
  return out;
}

Tensor ge(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ge_out(
      result, self, wrapped_scalar_tensor(other_));
}

/*=========================== le ==========================*/

Tensor& le_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::le_kernel_dpcpp(iter);
  return out;
}

Tensor le(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::le_out(result, self, other);
}

Tensor& le_out(Tensor& out, const Tensor& self, Scalar other_) {
  at::AtenIpexTypeXPU::le_out(out, self, wrapped_scalar_tensor(other_));
  return out;
}

Tensor le(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::le_out(
      result, self, wrapped_scalar_tensor(other_));
}

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

/*=========================== ne ==========================*/

Tensor& ne_out(Tensor& out, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  impl::ne_kernel_dpcpp(iter);
  return out;
}

Tensor ne(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ne_out(result, self, other);
}

Tensor& ne_out(Tensor& out, const Tensor& self, Scalar other_) {
  at::AtenIpexTypeXPU::ne_out(out, self, wrapped_scalar_tensor(other_));
  return out;
}

Tensor ne(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::ne_out(
      result, self, wrapped_scalar_tensor(other_));
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
