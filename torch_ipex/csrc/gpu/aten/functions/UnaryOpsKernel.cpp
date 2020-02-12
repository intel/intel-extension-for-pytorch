#include <limits>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <core/SYCL.h>
#include <functions/Loops.h>


DP_DEF_K1(bitwise_not);
DP_DEF_K1(logical_not);
DP_DEF_K1(neg);

namespace at { namespace native {

void bitwise_not_kernel_sycl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    sycl_kernel_for_tensor_iter<DP_K(bitwise_not)>(iter, [](bool a) -> bool {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_sycl", [&]() {
      sycl_kernel_for_tensor_iter<DP_K(bitwise_not)>(iter, [](scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

void logical_not_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_sycl", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_sycl", [&]() {
      sycl_kernel_for_tensor_iter<DP_K(logical_not, self_t)>(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

void neg_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, iter.dtype(), "neg_sycl", [&]() {
    sycl_kernel_for_tensor_iter<DP_K(neg)>(iter, [](scalar_t a) -> scalar_t {
      return -a;
    });
  });
}

}}

namespace at { namespace AtenIpexTypeDPCPP {

Tensor bitwise_not(const Tensor & self){
  Tensor result = at::empty({0}, self.options());
  return bitwise_not_out(result, self); 
}

Tensor & bitwise_not_(Tensor & self){
  return bitwise_not_out(self, self);
}

Tensor & bitwise_not_out(Tensor & out, const Tensor & self){
  auto iter = TensorIterator::unary_op(out, self,
    /*check_mem_overlap=*/true);
  at::native::bitwise_not_kernel_sycl(iter);
  #ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
  #endif
  return out;
}

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return logical_not_out(self, self);
}

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.build();
  at::native::logical_not_kernel_sycl(iter);
  return result;
}

Tensor neg(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return neg_out(result, self);
}

Tensor& neg_(Tensor& self) {
  return neg_out(self, self);
}

Tensor& neg_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  auto iter = TensorIterator::unary_op(result, self,
    /*check_internal_overlap=*/true);
  at::native::neg_kernel_sycl(iter);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at