#include <limits>
#include <c10/dpcpp/SYCL.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/dpcpp/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

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

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_sycl);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_sycl);
REGISTER_DISPATCH(neg_stub, &neg_kernel_sycl);

}}
