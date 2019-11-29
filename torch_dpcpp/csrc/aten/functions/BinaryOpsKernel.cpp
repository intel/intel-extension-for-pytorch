#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <functions/Loops.h>


namespace at { namespace native {

//Note: sycl compiler does not support uname type in template.
class SyclOpAdd{};
class SyclOpMul{};
class SyclOpDiv{};

static void add_kernel_sycl(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t> ();
    sycl_kernel_for_tensor_iter<SyclOpAdd>(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a + alpha * b;
        });
  });
}

static void sub_kernel_sycl(TensorIterator& iter, Scalar alpha_scalar) {
  return add_kernel_sycl(iter, -alpha_scalar);
}

static void mul_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "mul", [&]() {
    sycl_kernel_for_tensor_iter<SyclOpMul>(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * b;
        });
  });
}

static void div_kernel_sycl(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      sycl_kernel_for_tensor_iter<SyclOpDiv>(iter,
        [](scalar_t a, scalar_t b)-> scalar_t {
        return a / b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "div", [&]() {
      sycl_kernel_for_tensor_iter<SyclOpDiv>(iter,
        [](scalar_t a, scalar_t b)-> scalar_t {
        return a / b;
      });
    });
  }
}

}
}
