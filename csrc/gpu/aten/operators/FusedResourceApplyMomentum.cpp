#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <tensor/Tensor.h>
#include <torch/library.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

#include <aten/operators/MemoryAccess.h>

using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct ComputeFusedResourceApplyMomentumKernelFunctor {
  std::tuple<scalar_t, scalar_t> operator()(
      scalar_t weight_elem,
      scalar_t momentum_buffer_elem,
      scalar_t grad_elem) const {
    // mom_t = mom * self.momentum - grad * scaled_lr
    auto temp_momentum_buffer =
        momentum_buffer_elem * momentum - grad_elem * lr;
    if (nesterov) {
      // var_t = var + mom_t * self.momentum - grad * scaled_lr
      weight_elem += temp_momentum_buffer * momentum - grad_elem * lr;
    } else {
      // var_t = var + mom_t
      weight_elem += temp_momentum_buffer;
    }
    return std::tuple<scalar_t, scalar_t>(
        static_cast<scalar_t>(weight_elem),
        static_cast<scalar_t>(temp_momentum_buffer));
  }

  ComputeFusedResourceApplyMomentumKernelFunctor(
      float momentum_,
      float lr_,
      bool nesterov_)
      : momentum(momentum_), lr(lr_), nesterov(nesterov_) {}

 private:
  const float momentum;
  const float lr;
  const bool nesterov;
};

void ComputeFusedResourceApplyMomentumKernel(
    Tensor& weight,
    Tensor& momentum_buffer,
    const Tensor& grad,
    const float momentum,
    const float lr,
    const bool nesterov) {
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(weight)
                  .add_output(momentum_buffer)
                  .add_input(weight)
                  .add_input(momentum_buffer)
                  .add_input(grad)
                  .build();
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "ComputeFusedResourceApplyMomentumKernel",
      [&]() {
        ComputeFusedResourceApplyMomentumKernelFunctor<scalar_t> f(
            momentum, lr, nesterov);
        dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
      });
}
} // namespace impl

// The fusion for a new optimizer step function is implemented here
c10::optional<Tensor> fused_resource_apply_momentum(
    Tensor& param,
    Tensor& momentum_buffer,
    const Tensor& grad,
    const double momentum,
    const double lr,
    const bool nesterov) {
  RECORD_FUNCTION(
      "fused_resource_apply_momentum",
      std::vector<c10::IValue>(
          {param, momentum_buffer, grad, momentum, lr, nesterov}));
  const OptionalDeviceGuard device_guard(device_of(grad));
  auto momentum_value = static_cast<float>(momentum);
  auto lr_value = static_cast<float>(lr);

  // after inference, the model weight/grad in the next training epoch maybe
  // cached block, so to plain now if needed
  at::AtenIpexTypeXPU::to_plain_if_needed_(param);
  auto grad_value = at::AtenIpexTypeXPU::to_plain_if_needed(grad);
  impl::ComputeFusedResourceApplyMomentumKernel(
      param, momentum_buffer, grad_value, momentum_value, lr_value, nesterov);
  return param;
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fused_resource_apply_momentum",
      at::AtenIpexTypeXPU::fused_resource_apply_momentum,
      c10::DispatchKey::XPU);
}
} // namespace