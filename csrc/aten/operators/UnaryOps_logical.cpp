#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void logical_not_kernel(TensorIterator& iter) {
  // NOTE: We should not dispatch on types which aren't in below
  // ALL_TYPES_AND... Therefore, we add the check here.
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_dpcpp", [&]() {});

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> bool { return !a; });
      });
}

void signbit_kernel(TensorIteratorBase& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_dpcpp", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> bool {
          return !dpl::is_unsigned<scalar_t>::value && a < 0;
        });
      });
}

} // namespace impl

Tensor& logical_not_out(const Tensor& self, Tensor& result) {
  TensorIterator iter = TensorIteratorConfig()
                            .check_all_same_dtype(false)
                            .set_check_mem_overlap(true)
                            .add_output(result)
                            .add_input(self)
                            .build();
  impl::logical_not_kernel(iter);
  return result;
}

Tensor& signbit_out(const Tensor& self, Tensor& result) {
  if (self.dtype() == at::kBool) {
    result.fill_(false);
  } else {
    auto iter = TensorIterator::unary_op(result, self);
    impl::signbit_kernel(iter);
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
