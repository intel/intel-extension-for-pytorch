#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include <oneapi/dpl/tuple>
#include "Loops.h"
#include "comm/LoopsMeta.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void frexp_kernel_dpcpp(TensorIteratorBase& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      // The iter.dtype() here is the dtype of mantissa output.
      // It's a floating point type and must be the same as the input's dtype.
      iter.dtype(),
      "frexp_dpcpp",
      [&]() {
        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter, [=](scalar_t a) -> dpl::tuple<scalar_t, int32_t> {
              int32_t exponent;
              scalar_t mantissa = std::frexp(a, &exponent);
              return {mantissa, exponent};
            });
      });
}
} // namespace impl

std::tuple<Tensor&, Tensor&> frexp_out(
    const Tensor& self,
    Tensor& mantissa,
    Tensor& exponent) {
  // torch.frexp is implemented for floating-point dtypes for now,
  // should add support for integral dtypes in the future.
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "torch.frexp() only supports floating-point dtypes");

  TORCH_CHECK(
      mantissa.dtype() == self.dtype(),
      "torch.frexp() expects mantissa to have dtype ",
      self.dtype(),
      " but got ",
      mantissa.dtype());
  TORCH_CHECK(
      exponent.dtype() == at::kInt,
      "torch.frexp() expects exponent to have int dtype "
      "but got ",
      exponent.dtype());

  auto iter = TensorIteratorConfig()
                  .add_output(mantissa)
                  .add_output(exponent)
                  .add_input(self)
                  .check_all_same_dtype(false)
                  .set_check_mem_overlap(true)
                  .build();

  impl::frexp_kernel_dpcpp(iter);
  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}

std::tuple<Tensor, Tensor> frexp(const Tensor& self) {
  Tensor mantissa = at::empty_like(self);
  Tensor exponent = at::empty_like(self, self.options().dtype(at::kInt));

  frexp_out(mantissa, exponent, self);
  return std::tuple<Tensor, Tensor>(mantissa, exponent);
}

} // namespace AtenIpexTypeXPU
} // namespace at
