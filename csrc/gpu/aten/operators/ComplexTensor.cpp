#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

namespace at {
namespace impl {
void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK(
      (a.scalar_type() == kFloat || a.scalar_type() == kDouble) &&
          (b.scalar_type() == kFloat || b.scalar_type() == kDouble),
      "Expected both inputs to be Float or Double tensors but got ",
      a.scalar_type(),
      " and ",
      b.scalar_type());
}

void complex_check_dtype(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b) {
  complex_check_floating(a, b);
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "Expected object of scalar type ",
      a.scalar_type(),
      " but got scalar type ",
      b.scalar_type(),
      " for second argument");
  TORCH_CHECK(
      result.scalar_type() == toComplexType(a.scalar_type()),
      "Expected object of scalar type ",
      toComplexType(a.scalar_type()),
      " but got scalar type ",
      result.scalar_type(),
      " for argument 'out'");
}

void polar_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar", [&]() {
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
          return c10::complex<scalar_t>(
              a * Numerics<scalar_t>::cos(b), a * Numerics<scalar_t>::sin(b));
        });
  });
}
} // namespace impl

namespace AtenIpexTypeXPU {

Tensor& complex_out(const Tensor& real, const Tensor& imag, Tensor& result) {
  impl::complex_check_dtype(result, real, imag);
  auto iter = TensorIteratorConfig()
                  .add_output(result)
                  .add_input(real)
                  .add_input(imag)
                  .check_all_same_dtype(false)
                  .build();
  IPEX_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "complex_out", [&]() {
    dpcpp_kernel_for_tensor_iter(
        iter, [](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
          return c10::complex<scalar_t>(a, b);
        });
  });

  return result;
}

Tensor& polar_out(const Tensor& abs, const Tensor& angle, Tensor& out) {
  impl::complex_check_dtype(out, abs, angle);
  auto iter = TensorIteratorConfig()
                  .add_output(out)
                  .add_input(abs)
                  .add_input(angle)
                  .check_all_same_dtype(false)
                  .build();
  impl::polar_dpcpp(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
