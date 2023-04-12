#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& tanh_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  return unary_out_with_onednn_and_loops_bw<
      dnnl::algorithm::eltwise_tanh_use_dst_for_bwd>(
      TensorIterator::binary_op,
      grad_input,
      grad_output,
      output,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "tanh_backward_out",
            [&]() {
              dpcpp_kernel_for_tensor_iter(
                  iter, [](scalar_t output, scalar_t z) -> scalar_t {
                    return output * (scalar_t{1} - z * z);
                  });
            });
      });
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  auto grad_input = at::empty_like(grad_output);
  return at::AtenIpexTypeXPU::tanh_backward_out(
      grad_output, output, grad_input);
}

void atan2_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "atan2",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a, scalar_t b) -> scalar_t {
              return Numerics<scalar_t>::atan2(a, b);
            });
      });
}

Tensor& atan2_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_kernel(iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_float_op(result, self, other);
  atan2_kernel(iter);
  return iter.output();
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeXPU::atan2_out(self, other, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
