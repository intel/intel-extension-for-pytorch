#include <c10/dpcpp/SYCL.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/dpcpp/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Tanh.h>

DP_DEF_K1(tanh_backward);

namespace at { namespace native {

static void tanh_backward_sycl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "tanh_backward", [&]() {
    sycl_kernel_for_tensor_iter<DP_K(tanh_backward)>(iter, [](scalar_t z, scalar_t output) -> scalar_t {
      return output * (1. - z*z);
    });
  });
}

REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_sycl);

}} // namespace at::native

