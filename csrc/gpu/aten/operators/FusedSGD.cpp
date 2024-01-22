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
#include "Converter.h"
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

struct ComputeSplitSGDKernelFunctor {
  std::tuple<at::BFloat16, at::BFloat16> operator()(
      at::BFloat16 top_half_elem,
      at::BFloat16 tail_half_elem,
      at::BFloat16 grad_elem) const {
    float weight_elem = pack_bloat16_float(top_half_elem, tail_half_elem);
    auto grad_elem_fp32 = static_cast<float>(grad_elem);

    // d_p = d_p.add(p.master_weight, alpha=weight_decay)
    if (using_weight_decay) {
      grad_elem_fp32 += weight_elem * weight_decay_value;
    }

    // p.master_weight.add_(d_p, alpha=-group['lr'])
    auto res = static_cast<float>(weight_elem + grad_elem_fp32 * negative_lr);

    return unpack_float_bfloat16(res);
  }

  ComputeSplitSGDKernelFunctor(
      float weight_decay_value_,
      bool using_weight_decay_,
      float negative_lr_)
      : weight_decay_value(weight_decay_value_),
        using_weight_decay(using_weight_decay_),
        negative_lr(negative_lr_) {}

 private:
  float weight_decay_value;
  bool using_weight_decay;
  float negative_lr;
};

struct ComputeSplitSGDKernelFunctor2 {
  std::tuple<at::BFloat16, at::BFloat16, float> operator()(
      at::BFloat16 top_elem,
      at::BFloat16 tail_elem,
      at::BFloat16 grad_elem,
      float momentum_elem) const {
    float weight_elem = pack_bloat16_float(top_elem, tail_elem);
    // d_p = d_p.add(p, alpha=weight_decay)
    auto grad_elem_fp32 = static_cast<float>(grad_elem);

    auto temp_master_weight_value = weight_elem;

    // d_p = d_p.add(p.master_weight, alpha=weight_decay)
    if (using_weight_decay) {
      grad_elem_fp32 += temp_master_weight_value * weight_decay_value;
    }

    // 'momentum_buffer' in param_state,
    // param_state[momentum_buffer] has been created
    auto temp_momentum_buffer_value = grad_elem_fp32;
    if (momentum_buf_initialized) {
      // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
      temp_momentum_buffer_value = momentum_elem;
      temp_momentum_buffer_value = momentum_value * temp_momentum_buffer_value;
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

    std::tie(top_elem, tail_elem) = unpack_float_bfloat16(res);
    return std::tuple<at::BFloat16, at::BFloat16, float>(
        top_elem, tail_elem, static_cast<float>(temp_momentum_buffer_value));
  }

  ComputeSplitSGDKernelFunctor2(
      float weight_decay_value_,
      bool using_weight_decay_,
      float negative_lr_,
      float momentum_value_,
      float pre_dampening_,
      const bool nesterov_,
      const bool momentum_buf_initialized_)
      : weight_decay_value(weight_decay_value_),
        using_weight_decay(using_weight_decay_),
        negative_lr(negative_lr_),
        momentum_value(momentum_value_),
        pre_dampening(pre_dampening_),
        nesterov(nesterov_),
        momentum_buf_initialized(momentum_buf_initialized_) {}

 private:
  float weight_decay_value;
  bool using_weight_decay;
  float negative_lr;
  float momentum_value;
  float pre_dampening;
  const bool nesterov;
  const bool momentum_buf_initialized;
};

// for SplitSGD Kernel, both top_half, grad, tail_half are at::BFloat16
static void ComputeSplitSGDKernel(
    Tensor& top_half,
    Tensor& grad,
    Tensor& momentum_buffer,
    Tensor& tail_half,
    const double weight_decay,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    const bool momentum_buf_initialized) {
  TORCH_CHECK(
      top_half.scalar_type() == at::kBFloat16,
      "ComputeSplitSGDKernel: expect top_half to be at::BFloat16");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "ComputeSplitSGDKernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      !momentum_buffer.defined() || momentum_buffer.scalar_type() == at::kFloat,
      "ComputeSplitSGDKernel: expect momentum_buffer to be float32");
  TORCH_CHECK(
      tail_half.scalar_type() == at::kBFloat16,
      "ComputeSplitSGDKernel: expect tail_half to be at::kBFloat16");

  auto using_momentum = bool(momentum);
  auto using_weight_decay = bool(weight_decay);
  auto weight_decay_value = static_cast<float>(weight_decay);
  auto momentum_value = static_cast<float>(momentum);
  auto pre_dampening = static_cast<float>(1.0 - dampening);
  auto negative_lr = static_cast<float>((-1.0) * lr);

  if (!using_momentum) {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .check_all_same_dtype(false)
                                  .add_output(top_half)
                                  .add_output(tail_half)
                                  .add_input(top_half)
                                  .add_input(tail_half)
                                  .add_input(grad)
                                  .build();
    ComputeSplitSGDKernelFunctor f(
        weight_decay_value, using_weight_decay, negative_lr);
    dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
  } else {
    // use momentum
    at::TensorIterator iter = TensorIteratorConfig()
                                  .check_all_same_dtype(false)
                                  .add_output(top_half)
                                  .add_output(tail_half)
                                  .add_output(momentum_buffer)
                                  .add_input(top_half)
                                  .add_input(tail_half)
                                  .add_input(grad)
                                  .add_input(momentum_buffer)
                                  .build();
    ComputeSplitSGDKernelFunctor2 f(
        weight_decay_value,
        using_weight_decay,
        negative_lr,
        momentum_value,
        pre_dampening,
        nesterov,
        momentum_buf_initialized);
    dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
  }
}

template <typename scalar_t>
struct ComputeSGDKernelMasterWeightFunctor {
  std::tuple<scalar_t, float> operator()(
      scalar_t weight_elem,
      scalar_t grad_elem,
      float master_weight_elem) const {
    auto grad_elem_fp32 = static_cast<float>(grad_elem);

    // d_p = d_p.add(p.master_weight, alpha=weight_decay)
    if (using_weight_decay) {
      grad_elem_fp32 += master_weight_elem * weight_decay_value;
    }

    // p.master_weight.add_(d_p, alpha=-group['lr'])
    auto res =
        static_cast<float>(master_weight_elem + grad_elem_fp32 * negative_lr);

    return std::tuple<scalar_t, float>(static_cast<scalar_t>(res), res);
  }

  ComputeSGDKernelMasterWeightFunctor(
      float weight_decay_value_,
      bool using_weight_decay_,
      float negative_lr_)
      : weight_decay_value(weight_decay_value_),
        using_weight_decay(using_weight_decay_),
        negative_lr(negative_lr_) {}

 private:
  float weight_decay_value;
  bool using_weight_decay;
  float negative_lr;
};

template <typename scalar_t>
struct ComputeSGDKernelMasterWeightFunctor2 {
  std::tuple<scalar_t, float, scalar_t> operator()(
      scalar_t weight_elem,
      scalar_t grad_elem,
      float master_weight_elem,
      scalar_t momentum_elem) const {
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
      temp_momentum_buffer_value = momentum_elem;
      temp_momentum_buffer_value = momentum_value * temp_momentum_buffer_value;
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
  }

  ComputeSGDKernelMasterWeightFunctor2(
      float weight_decay_value_,
      bool using_weight_decay_,
      float negative_lr_,
      float momentum_value_,
      float pre_dampening_,
      const bool nesterov_,
      const bool momentum_buf_initialized_)
      : weight_decay_value(weight_decay_value_),
        using_weight_decay(using_weight_decay_),
        negative_lr(negative_lr_),
        momentum_value(momentum_value_),
        pre_dampening(pre_dampening_),
        nesterov(nesterov_),
        momentum_buf_initialized(momentum_buf_initialized_) {}

 private:
  float weight_decay_value;
  bool using_weight_decay;
  float negative_lr;
  float momentum_value;
  float pre_dampening;
  const bool nesterov;
  const bool momentum_buf_initialized;
};

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
    ComputeSGDKernelMasterWeightFunctor<scalar_t> f(
        weight_decay_value, using_weight_decay, negative_lr);
    dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
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
    ComputeSGDKernelMasterWeightFunctor2<scalar_t> f(
        weight_decay_value,
        using_weight_decay,
        negative_lr,
        momentum_value,
        pre_dampening,
        nesterov,
        momentum_buf_initialized);
    dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
  }
}

template <typename scalar_t>
struct ComputeSGDKernelFunctor {
  scalar_t operator()(scalar_t weight_elem, scalar_t grad_elem) const {
    if (using_weight_decay) {
      grad_elem += weight_elem * weight_decay_value;
    }
    return weight_elem + grad_elem * negative_lr;
  }

  ComputeSGDKernelFunctor(
      bool using_weight_decay,
      float weight_decay_value,
      float negative_lr)
      : using_weight_decay(using_weight_decay),
        weight_decay_value(weight_decay_value),
        negative_lr(negative_lr) {}

 private:
  bool using_weight_decay;
  float weight_decay_value;
  float negative_lr;
};

template <typename scalar_t>
struct ComputeSGDKernelFunctor2 {
  std::tuple<scalar_t, scalar_t> operator()(
      scalar_t weight_elem,
      scalar_t grad_elem,
      scalar_t momentum_elem) const {
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
      temp_momentum_buffer_value = momentum_value * temp_momentum_buffer_value;
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

    return std::tuple<scalar_t, scalar_t>(res, temp_momentum_buffer_value);
  }

  ComputeSGDKernelFunctor2(
      float weight_decay_value_,
      bool using_weight_decay_,
      float negative_lr_,
      float momentum_value_,
      float pre_dampening_,
      const bool nesterov_,
      const bool momentum_buf_initialized_)
      : weight_decay_value(weight_decay_value_),
        using_weight_decay(using_weight_decay_),
        negative_lr(negative_lr_),
        momentum_value(momentum_value_),
        pre_dampening(pre_dampening_),
        nesterov(nesterov_),
        momentum_buf_initialized(momentum_buf_initialized_) {}

 private:
  float weight_decay_value;
  bool using_weight_decay;
  float negative_lr;
  float momentum_value;
  float pre_dampening;
  const bool nesterov;
  const bool momentum_buf_initialized;
};

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
    ComputeSGDKernelFunctor<scalar_t> f(
        using_weight_decay, weight_decay_value, negative_lr);
    dpcpp_kernel_for_tensor_iter(iter, f);
  } else {
    // use momentum
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(weight)
                                  .add_output(momentum_buffer)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(momentum_buffer)
                                  .build();
    ComputeSGDKernelFunctor2<scalar_t> f(
        weight_decay_value,
        using_weight_decay,
        negative_lr,
        momentum_value,
        pre_dampening,
        nesterov,
        momentum_buf_initialized);
    dpcpp_kernel_multiple_outputs_for_tensor_iter(iter, f);
  }
}

} // namespace impl

// [watch out] This is used for fusion optimizer SGD step function
// For datatype, top_weight: top master weight and fp32 weight(some layer no
// need cast), and maybe bf16 for split sgd, it will repersent the top half.
// grad: bf16/fp32 grad, bf16 grad from casted layer, fp32 grad
// from no casted layer momentum_buffer: be none for first iter.
// weight: bf16 weight(mapped to fp32 master weight or the bottom half of split
// weight) and empty tensor(empty means no need casted latey's weight)
c10::optional<at::Tensor> sgd_fused_step(
    at::Tensor& top_weight,
    const at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    at::Tensor& accompany_weight,
    double momentum,
    double lr,
    double weight_decay,
    double dampening,
    bool nesterov) {
  RECORD_FUNCTION(
      "sgd_fused_step",
      std::vector<c10::IValue>(
          {top_weight, grad, momentum_buffer_, accompany_weight}));
  const OptionalDeviceGuard device_guard(device_of(top_weight));

  // after inference, the model weight in the next training epoch maybe cached
  // block, so to plain now if needed
  top_weight = to_plain_if_needed_(top_weight);
  Tensor grad_ = to_plain_if_needed(grad);

  c10::ScalarType momentum_buffer_type = top_weight.scalar_type();
  if (top_weight.scalar_type() == at::ScalarType::BFloat16) {
    momentum_buffer_type = at::ScalarType::Float;
  }

  at::Tensor momentum_buffer;
  bool momentum_buf_initialized;
  if (momentum) {
    if (!momentum_buffer_.has_value()) {
      momentum_buffer = at::empty_like(
          top_weight, top_weight.options().dtype(momentum_buffer_type));
      momentum_buf_initialized = false;
    } else {
      momentum_buffer = momentum_buffer_.value();
      momentum_buf_initialized = true;
    }
  }

  // for split sgd, here both weight and grad is bf16, and weight is top half,
  // accompany_weight is bottom half
  if (at::ScalarType::BFloat16 == top_weight.scalar_type() &&
      at::ScalarType::BFloat16 == grad.scalar_type() &&
      at::ScalarType::BFloat16 == accompany_weight.scalar_type()) {
    impl::ComputeSplitSGDKernel(
        top_weight,
        grad_,
        momentum_buffer,
        accompany_weight,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        lr,
        momentum_buf_initialized);
    if (!momentum) {
      return c10::nullopt;
    } else {
      return momentum_buffer;
    }
  }

  // master weight mode, top_weight contains fp32 master weight, weight is
  // bf16 weight, grad is bf16
  if (accompany_weight.numel()) {
    // after inference, the model weight in the next training epoch maybe cached
    // block, so to plain now if needed
    accompany_weight = to_plain_if_needed_(accompany_weight);

    auto memory_format = accompany_weight.suggest_memory_format();
    top_weight = top_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    accompany_weight = accompany_weight.contiguous(memory_format);
    grad_ = grad_.contiguous(memory_format);

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        accompany_weight.scalar_type(),
        "fusion_sgd_with_master_weight",
        [&] {
          impl::ComputeSGDKernelMasterWeight<scalar_t>(
              top_weight,
              grad_,
              momentum_buffer,
              accompany_weight,
              weight_decay,
              momentum,
              dampening,
              nesterov,
              lr,
              momentum_buf_initialized);
        });
  } else {
    auto memory_format = top_weight.suggest_memory_format();
    top_weight = top_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    grad_ = grad_.contiguous(memory_format);

    // normal mode, param_ is fp32 or fp64
    IPEX_DISPATCH_FLOATING_TYPES(
        top_weight.scalar_type(), "sgd_fused_step", [&] {
          impl::ComputeSGDKernel<scalar_t>(
              top_weight,
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