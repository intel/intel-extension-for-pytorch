#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/Pairwise.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K1(bitwise_not);
void bitwise_not_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter<DPCPP_K(bitwise_not)>(
        iter, [](bool a) -> bool { return !a; });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_dpcpp", [&]() {
      dpcpp_kernel_for_tensor_iter<DPCPP_K(bitwise_not)>(
          iter, [](scalar_t a) -> scalar_t { return ~a; });
    });
  }
}

DPCPP_DEF_K1(logical_not);
void logical_not_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, iter.dtype(1), "logical_not_dpcpp", [&]() {
        using self_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            kBool, kHalf, iter.dtype(0), "logical_not_dpcpp", [&]() {
              dpcpp_kernel_for_tensor_iter<DPCPP_K(logical_not, self_t)>(
                  iter, [](self_t a) -> scalar_t {
                    return static_cast<scalar_t>(!a);
                  });
            });
      });
}

} // namespace impl

Tensor bitwise_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::bitwise_not_out(result, self);
}

Tensor& bitwise_not_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::bitwise_not_out(self, self);
}

Tensor& bitwise_not_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(
      out,
      self,
      /*check_mem_overlap=*/true);
  impl::bitwise_not_kernel_dpcpp(iter);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
#endif
  return out;
}

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeDPCPP::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::logical_not_out(self, self);
}

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.build();
  impl::logical_not_kernel(iter);
  return result;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
