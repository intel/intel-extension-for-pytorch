#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include "comm/ATDispatch.h"
#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

DPCPP_DEF_K1(DPCPPOpHardTanh);
Tensor& hardtanh_out(
    Tensor& out,
    const Tensor& self,
    Scalar min_val,
    Scalar max_val) {
  checkBackend("hardtanh", out, self.options().backend());
  // Compare the norm and maxnorm value.
  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(out)
  .add_input(self)
  .build();

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardtanh",
      [&]() {
        scalar_t min_ = min_val.to<scalar_t>();
        scalar_t max_ = max_val.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardTanh)>(
            iter, [=](scalar_t x) -> scalar_t {
              if (x < min_)
                return min_;
              else if (x > max_)
                return max_;
              else
                return x;
            });
      });

  return out;
}

Tensor hardtanh(const Tensor& self, Scalar min_val, Scalar max_val) {
  TORCH_CHECK(!self.is_sparse(), "hardtanh(dpcpp_sparse) is not supported.");
  Tensor result = at::empty(self.sizes(), self.options());
  at::AtenIpexTypeXPU::hardtanh_out(result, self, min_val, max_val);
  return result;
}

Tensor& hardtanh_(Tensor& self, Scalar min_val, Scalar max_val) {
  return at::AtenIpexTypeXPU::hardtanh_out(self, self, min_val, max_val);
}

DPCPP_DEF_K1(DPCPPOpHardTanhBackward);
Tensor& hardtanh_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar min_val,
    Scalar max_val) {
  checkBackend(
      "hardtanh_backward", {grad_input, grad_output}, self.options().backend());
  // Compare the norm and maxnorm value.
  auto iter = TensorIteratorConfig()
  .set_check_mem_overlap(true)
  .add_output(grad_input)
  .add_input(grad_output)
  .add_input(self)
  .build();

  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "hardtanh_backward", [&]() {
        auto min_ = min_val.to<scalar_t>();
        auto max_ = max_val.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardTanhBackward)>(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              if (x <= min_ || x >= max_)
                return 0;
              else
                return grad_output;
            });
      });

  return grad_input;
}

Tensor hardtanh_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar min_val,
    Scalar max_val) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeXPU::hardtanh_backward_out(
      grad_input, grad_output, self, min_val, max_val);
}
} // namespace AtenIpexTypeXPU
} // namespace at
