#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <torch/library.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include <ATen/native/ForeachUtils.h>
#include <aten/operators/MemoryAccess.h>
#include "ATen/OpMathType.h"
#include "ForeachFunctors.h"
#include "FusedFunctors.h"
#include "Loops.h"
#include "LoopsTemplates.h"
#include "MultiTensorApply.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct ema_fused_impl_functor {
  scalar_t operator()(scalar_t model, scalar_t ema, scalar_t decay) const {
    ema = ema * decay + (1 - decay) * model;
    return ema;
  }
};

template <typename scalar_t>
void ema_fused_impl(Tensor model_param, Tensor ema_param, Tensor decay) {
  at::TensorIterator iter = TensorIteratorConfig()
                                .add_output(ema_param)
                                .add_input(model_param)
                                .add_input(ema_param)
                                .add_input(decay)
                                .build();
  ema_fused_impl_functor<scalar_t> f;
  dpcpp_kernel_for_tensor_iter(iter, f);
}

void ema_fused_step(
    TensorList model_params,
    TensorList ema_params,
    Tensor decay) {
  for (int i = 0; i < model_params.size(); i++) {
    Tensor model_param = model_params[i];
    Tensor ema_param = ema_params[i];
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        model_params[0].scalar_type(),
        "ema_fused_step_kernel",
        [&]() { ema_fused_impl<scalar_t>(model_param, ema_param, decay); });
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "ema_fused_step",
      at::AtenIpexTypeXPU::ema_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace
