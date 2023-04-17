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

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
static void ComputeSGDKernelMasterWeight(
    Tensor& master_weight,
    Tensor& grad,
    Tensor& momentum_buffer,
    Tensor& weight,
    const double weight_decay,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    const bool momentum_buf_initialized) {
  TORCH_CHECK(
      master_weight.scalar_type() == at::kFloat,
      "ComputeSGDKernelMasterWeight: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "ComputeSGDKernelMasterWeight: expect grad to be at::BFloat16");
  TORCH_CHECK(
      !momentum_buffer.defined() || momentum_buffer.scalar_type() == at::kFloat,
      "ComputeSGDKernelMasterWeight: expect momentum_buffer to be float32");
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16,
      "ComputeSGDKernelMasterWeight: expect param to be at::kBFloat16");

  auto using_momentum = bool(momentum);
  auto using_weight_decay = bool(weight_decay);
  auto weight_decay_value = static_cast<float>(weight_decay);
  auto momentum_value = static_cast<float>(momentum);
  auto pre_dampening = static_cast<float>(1.0 - dampening);
  auto negative_lr = static_cast<float>((-1.0) * lr);

  if (!using_momentum) {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .check_all_same_dtype(false)
                                  .add_output(weight)
                                  .add_output(master_weight)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(master_weight)
                                  .build();

    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t weight_elem,
            scalar_t grad_elem,
            float master_weight_elem) -> std::tuple<scalar_t, float> {
          auto grad_elem_fp32 = static_cast<float>(grad_elem);

          // d_p = d_p.add(p.master_weight, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elem_fp32 += master_weight_elem * weight_decay_value;
          }

          // p.master_weight.add_(d_p, alpha=-group['lr'])
          auto res = static_cast<float>(
              master_weight_elem + grad_elem_fp32 * negative_lr);

          return std::tuple<scalar_t, float>(static_cast<scalar_t>(res), res);
        });
  } else {
    // use momentum
    at::TensorIterator iter = TensorIteratorConfig()
                                  .check_all_same_dtype(false)
                                  .add_output(weight)
                                  .add_output(master_weight)
                                  .add_output(momentum_buffer)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(master_weight)
                                  .add_input(momentum_buffer)
                                  .build();
    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t weight_elem,
            scalar_t grad_elem,
            float master_weight_elem,
            scalar_t momentum_elem) -> std::tuple<scalar_t, float, scalar_t> {
          // d_p = d_p.add(p, alpha=weight_decay)
          auto grad_elem_fp32 = static_cast<float>(grad_elem);

          auto temp_master_weight_value = master_weight_elem;

          // d_p = d_p.add(p.master_weight, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elem_fp32 += temp_master_weight_value * weight_decay_value;
          }

          // 'momentum_buffer' in param_state,
          // param_state[momentum_buffer] has been created
          auto temp_momentum_buffer_value = grad_elem_fp32;
          if (momentum_buf_initialized) {
            // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            temp_momentum_buffer_value = master_weight_elem;
            temp_momentum_buffer_value =
                momentum_value * temp_momentum_buffer_value;
            temp_momentum_buffer_value += grad_elem_fp32 * pre_dampening;
          }

          // nesterov
          if (nesterov) {
            // d_p = d_p.add(buf, alpha=momentum)
            grad_elem_fp32 += momentum_value * temp_momentum_buffer_value;
          } else {
            // d_p = buf
            grad_elem_fp32 = temp_momentum_buffer_value;
          }

          // p.master_weight.add_(d_p, alpha=-group['lr'])
          auto res = static_cast<float>(
              temp_master_weight_value + grad_elem_fp32 * negative_lr);
          return std::tuple<scalar_t, float, scalar_t>(
              static_cast<scalar_t>(res),
              res,
              static_cast<float>(temp_momentum_buffer_value));
        });
  }
}

template <typename scalar_t>
static void ComputeSGDKernel(
    Tensor& weight,
    Tensor& grad,
    Tensor& momentum_buffer,
    const double weight_decay,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    const bool momentum_buf_initialized) {
  TORCH_CHECK(
      weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kDouble,
      "ComputeSGDKernel: expect param to be at::kFloat or at::kDouble");
  TORCH_CHECK(
      grad.scalar_type() == at::kFloat || grad.scalar_type() == at::kDouble,
      "ComputeSGDKernel: expect grad to be at::kFloat or at::kDouble");
  TORCH_CHECK(
      !momentum_buffer.defined() ||
          (momentum_buffer.scalar_type() == at::kFloat ||
           momentum_buffer.scalar_type() == at::kDouble),
      "ComputeSGDKernel: expect momentum_buffer to be at::kFloat or at::kDouble");

  auto using_momentum = bool(momentum);
  auto using_weight_decay = bool(weight_decay);
  auto weight_decay_value = static_cast<float>(weight_decay);
  auto momentum_value = static_cast<float>(momentum);
  auto pre_dampening = static_cast<float>(1.0 - dampening);
  auto negative_lr = static_cast<float>((-1.0) * lr);

  if (!using_momentum) {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(weight)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .build();

    dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t weight_elem, scalar_t grad_elem) -> scalar_t {
          if (using_weight_decay) {
            grad_elem += weight_elem * weight_decay_value;
          }
          return weight_elem + grad_elem * negative_lr;
        });
  } else {
    // use momentum
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(weight)
                                  .add_output(momentum_buffer)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(momentum_buffer)
                                  .build();
    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t weight_elem,
            scalar_t grad_elem,
            scalar_t momentum_elem) -> std::tuple<scalar_t, scalar_t> {
          // d_p = d_p.add(p, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elem += weight_elem * weight_decay_value;
          }

          // 'momentum_buffer' in param_state,
          // param_state[momentum_buffer] has been created
          auto temp_momentum_buffer_value = grad_elem;
          if (momentum_buf_initialized) {
            // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            temp_momentum_buffer_value = momentum_elem;
            temp_momentum_buffer_value =
                momentum_value * temp_momentum_buffer_value;
            temp_momentum_buffer_value += grad_elem * pre_dampening;
          }

          // nesterov
          if (nesterov) {
            // d_p = d_p.add(buf, alpha=momentum)
            grad_elem += momentum_value * temp_momentum_buffer_value;
          } else {
            // d_p = buf
            grad_elem = temp_momentum_buffer_value;
          }

          // p.add_(d_p, alpha=-group['lr'])
          auto res = weight_elem + grad_elem * negative_lr;

          return std::tuple<scalar_t, scalar_t>(
              res, temp_momentum_buffer_value);
        });
  }
}

} // namespace impl

// [watch out] This is used for fusion optimizer SGD step function
// For datatype, fp32_weight: fp32 master weight and fp32 weight(some layer no
// need cast). grad: bf16/fp32 grad, bf16 grad from casted layer, fp32 grad
// from no casted layer momentum_buffer: be none for first iter. weight: bf16
// weight(mapped to fp32 master weight) and empty tensor(empty means no need
// casted latey's weight)
c10::optional<at::Tensor> sgd_fused_step(
    at::Tensor& fp32_weight,
    const at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    at::Tensor& weight,
    double momentum,
    double lr,
    double weight_decay,
    double dampening,
    bool nesterov) {
  RECORD_FUNCTION(
      "sgd_fused_step",
      std::vector<c10::IValue>({fp32_weight, grad, momentum_buffer_, weight}));
  const OptionalDeviceGuard device_guard(device_of(fp32_weight));

  // after inference, the model weight in the next training epoch maybe cached
  // block, so to plain now if needed
  fp32_weight = to_plain_if_needed_(fp32_weight);
  Tensor grad_ = to_plain_if_needed(grad);

  at::Tensor momentum_buffer;
  bool momentum_buf_initialized;
  if (momentum) {
    if (!momentum_buffer_.has_value()) {
      momentum_buffer = at::empty_like(fp32_weight);
      momentum_buf_initialized = false;
    } else {
      momentum_buffer = momentum_buffer_.value();
      momentum_buf_initialized = true;
    }
  }

  // master weight mode, fp32_weight contains fp32 master weight, weight is
  // bf16 weight, grad is bf16
  if (weight.numel()) {
    // after inference, the model weight in the next training epoch maybe cached
    // block, so to plain now if needed
    weight = to_plain_if_needed_(weight);

    auto memory_format = weight.suggest_memory_format();
    fp32_weight = fp32_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    weight = weight.contiguous(memory_format);
    grad_ = grad_.contiguous(memory_format);

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "fusion_sgd_with_master_weight",
        [&] {
          impl::ComputeSGDKernelMasterWeight<scalar_t>(
              fp32_weight,
              grad_,
              momentum_buffer,
              weight,
              weight_decay,
              momentum,
              dampening,
              nesterov,
              lr,
              momentum_buf_initialized);
        });
  } else {
    auto memory_format = fp32_weight.suggest_memory_format();
    fp32_weight = fp32_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    grad_ = grad_.contiguous(memory_format);

    // normal mode, param_ is fp32 or fp64
    IPEX_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "sgd_fused_step", [&] {
      impl::ComputeSGDKernel<scalar_t>(
          fp32_weight,
          grad_,
          momentum_buffer,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          momentum_buf_initialized);
    });
  }

  if (!momentum) {
    return c10::nullopt;
  } else {
    return momentum_buffer;
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "sgd_fused_step",
      at::AtenIpexTypeXPU::sgd_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace
