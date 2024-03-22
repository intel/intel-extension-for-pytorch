#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct logical_not_kernel_functor {
  bool operator()(scalar_t a) const {
    return !a;
  }
};

void logical_not_kernel(TensorIterator& iter) {
  // NOTE: We should not dispatch on types which aren't in below
  // ALL_TYPES_AND... Therefore, we add the check here.
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_dpcpp", [&]() {});

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_dpcpp", [&]() {
        logical_not_kernel_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

template <typename scalar_t>
struct signbit_kernel_functor {
  using opmath_t = at::opmath_type<scalar_t>;
  bool operator()(scalar_t a) const {
    return std::signbit(opmath_t{a});
  }
};

template <typename scalar_t>
struct signbit_kernel_int_type_functor {
  bool operator()(scalar_t a) const {
    return !std::is_unsigned<scalar_t>::value && a < 0;
  }
};

void signbit_kernel(TensorIteratorBase& iter) {
  if (at::isIntegralType(iter.input_dtype(), /*includeBool=*/false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.input_dtype(), "signbit_dpcpp", [&]() {
      signbit_kernel_int_type_functor<scalar_t> f;
      dpcpp_kernel_for_tensor_iter(iter, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        ScalarType::Half,
        iter.input_dtype(),
        "signbit_dpcpp",
        [&]() {
          signbit_kernel_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
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
    TensorIterator iter;
    iter.build_borrowing_unary_force_boolean_op(result, self);
    impl::signbit_kernel(iter);
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
