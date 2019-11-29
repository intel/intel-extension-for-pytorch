#include <c10/dpcpp/SYCL.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>

#include <functions/Loops.h>


DP_DEF_K1(addcmul);
DP_DEF_K1(addcdiv);

namespace at { namespace native {

static void addcmul_sycl_kernel(TensorIterator &iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "addcmul_sycl", [&]() {
    auto alpha = value.to<scalar_t>();
    sycl_kernel_for_tensor_iter<DP_K(addcmul)>(iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a + alpha * b * c;
    });
  });
}

static void addcdiv_sycl_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "addcdiv_sycl", [&]() {
    auto alpha = value.to<scalar_t>();
    sycl_kernel_for_tensor_iter<DP_K(addcdiv)>(iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a + alpha * (b / c);
    });
  });
}

REGISTER_DISPATCH(addcmul_stub, &addcmul_sycl_kernel);
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_sycl_kernel);

}} // namespace at::native

