#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
// #include <ATen/native/Tanh.h>

#include <core/SYCL.h>
#include <functions/Loops.h>


DP_DEF_K1(tanh_backward);

namespace at { namespace native {

static void tanh_backward_sycl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "tanh_backward", [&]() {
    sycl_kernel_for_tensor_iter<DP_K(tanh_backward)>(iter, [](scalar_t z, scalar_t output) -> scalar_t {
      return output * (1. - z*z);
    });
  });
}

}} // namespace at::native

