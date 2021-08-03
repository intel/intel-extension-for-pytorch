#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/AtenIpexTypeXPU.h>
#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K1(bitwise_not);
void bitwise_not_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter<DPCPP_K(bitwise_not)>(
        iter, [](bool a) -> bool { return !a; });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_dpcpp", [&]() {
      dpcpp_kernel_for_tensor_iter<DPCPP_K(bitwise_not)>(
          iter, [](scalar_t a) -> scalar_t { return ~a; });
    });
  }
}

DPCPP_DEF_K1(logical_not);
void logical_not_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, iter.dtype(1), "logical_not_dpcpp", [&]() {
        using self_t = scalar_t;
        IPEX_DISPATCH_ALL_TYPES_AND2(
            kBool, kHalf, iter.dtype(0), "logical_not_dpcpp", [&]() {
              dpcpp_kernel_for_tensor_iter<DPCPP_K(logical_not, self_t)>(
                  iter, [](self_t a) -> scalar_t {
                    return static_cast<scalar_t>(!a);
                  });
            });
      });
}

template <typename...>
class frac_kernel_class {};
void frac_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(at::kHalf, iter.dtype(), "frac_xpu", [&]() {
    dpcpp_kernel_for_tensor_iter<frac_kernel_class<scalar_t>>(
        iter, [](scalar_t a) -> scalar_t {
          return a - Numerics<scalar_t>::trunc(a);
        });
  });
}

} // namespace impl

Tensor bitwise_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::bitwise_not_out(result, self);
}

Tensor& bitwise_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::bitwise_not_out(self, self);
}

Tensor& bitwise_not_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  impl::bitwise_not_kernel_dpcpp(iter);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
#endif
  return out;
}

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeXPU::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::logical_not_out(self, self);
}

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter = TensorIteratorConfig()
                            .check_all_same_dtype(false)
                            .set_check_mem_overlap(true)
                            .add_output(result)
                            .add_input(self)
                            .build();
  impl::logical_not_kernel(iter);
  return result;
}

Tensor& frac_out(Tensor& result, const Tensor& self) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::frac_kernel(iter);
  return result;
}

Tensor frac(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::frac_out(result, self);
}

Tensor& frac_(Tensor& self) {
  return at::AtenIpexTypeXPU::frac_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
