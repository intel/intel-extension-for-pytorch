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

template <int vec_size, typename scalar_t>
void launch_vec_kernel_LARSMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& w_norm,
    Tensor& g_norm,
    const double weight_decay,
    const bool momentum_buf_initialized,
    Tensor& momentum_buffer,
    const double momentum,
    const double eeta,
    const double eps,
    const double lr,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  float* master_weight_ptr = master_weight.data_ptr<float>();
  scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
  scalar_t* grad_ptr = grad.data_ptr<scalar_t>();
  float* momentum_buffer_ptr =
      momentum_buffer.defined() ? momentum_buffer.data_ptr<float>() : nullptr;

  using accscalar_t = acc_type<scalar_t>;
  using vec_mw_t = at::native::Memory::aligned_vector_loop<float, vec_size>;
  using vec_w_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using vec_g_t = vec_w_t;

  vec_mw_t* master_weight_vec = reinterpret_cast<vec_mw_t*>(master_weight_ptr);
  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_g_t* grad_vec = reinterpret_cast<vec_g_t*>(grad_ptr);

  auto using_momentum = bool(momentum);
  auto using_weight_decay = bool(weight_decay);
  auto weight_decay_value = static_cast<accscalar_t>(weight_decay);
  auto momentum_value = static_cast<accscalar_t>(momentum);
  auto negative_lr = static_cast<accscalar_t>((-1.0) * lr);
  auto eps_value = static_cast<accscalar_t>(eps);
  auto eeta_value = static_cast<accscalar_t>(eeta);

  auto w_norm_ptr = w_norm.data_ptr<float>();
  auto g_norm_ptr = g_norm.data_ptr<scalar_t>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      // norm op will output one value tensor
      auto w_norm_value = w_norm_ptr[0];
      auto g_norm_value = static_cast<accscalar_t>(g_norm_ptr[0]);

      // eeta * w_norm / (g_norm + weight_decay * w_norm + eps)
      auto lars_lr = negative_lr;
      if ((w_norm_value > 0.0) && (g_norm_value > 0.0)) {
        lars_lr = negative_lr *
            (eeta * w_norm_value /
             (g_norm_value + weight_decay * w_norm_value + eps_value));
      }

      if (!using_momentum) {
        // vector value
        vec_mw_t master_weight_value = master_weight_vec[id];
        vec_w_t temp_weight;
        vec_mw_t temp_master_weight;

#pragma unroll(vec_size)
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          auto grad_elm = static_cast<float>(grad_vec[id][v_index]);
          auto temp_master_weight_value = master_weight_value[v_index];

          // d_p = d_p.add(p.master_weight, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elm += temp_master_weight_value * weight_decay_value;
          }

          // p.master_weight.add_(d_p, alpha=-group['lr'])
          auto res = temp_master_weight_value + grad_elm * lars_lr;
          temp_master_weight[v_index] = static_cast<float>(res);

          // p.data.copy_(p.master_weight.data)
          temp_weight[v_index] = static_cast<scalar_t>(res);
        }

        // write back
        master_weight_vec[id] = temp_master_weight;
        weight_vec[id] = temp_weight;
      } else {
        // momentum != 0
        // vector value
        vec_mw_t master_weight_value = master_weight_vec[id];
        vec_mw_t* momentum_buffer_vec =
            reinterpret_cast<vec_mw_t*>(momentum_buffer_ptr);

        // momentum buffer vector value
        vec_mw_t momentum_buffer_value = momentum_buffer_vec[id];
        vec_w_t temp_weight;
        vec_mw_t temp_master_weight;
        vec_mw_t temp_momentum_buffer;

#pragma unroll(vec_size)
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          auto grad_elm = static_cast<float>(grad_vec[id][v_index]);
          auto temp_master_weight_value = master_weight_value[v_index];

          // d_p = d_p.add(p.master_weight, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elm += temp_master_weight_value * weight_decay_value;
          }

          // 'momentum_buffer' in param_state,
          // param_state[momentum_buffer] has been created
          auto temp_momentum_buffer_value = grad_elm;
          if (momentum_buf_initialized) {
            // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            temp_momentum_buffer_value = momentum_buffer_value[v_index];
            temp_momentum_buffer_value =
                momentum_value * temp_momentum_buffer_value;
            temp_momentum_buffer_value += grad_elm;
          }
          temp_momentum_buffer[v_index] =
              static_cast<float>(temp_momentum_buffer_value);

          // d_p = buf
          grad_elm = temp_momentum_buffer_value;

          // p.master_weight.add_(d_p, alpha=-group['lr'])
          auto res =
              static_cast<float>(temp_master_weight_value + grad_elm * lars_lr);
          temp_master_weight[v_index] = res;

          // p.data.copy_(p.master_weight.data)
          temp_weight[v_index] = static_cast<scalar_t>(res);
        }

        // write back
        master_weight_vec[id] = temp_master_weight;
        weight_vec[id] = temp_weight;
        momentum_buffer_vec[id] = temp_momentum_buffer;
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <int vec_size>
void launch_vec_kernel_LARS(
    Tensor& weight,
    Tensor& grad,
    Tensor& w_norm,
    Tensor& g_norm,
    const double weight_decay,
    const bool momentum_buf_initialized,
    Tensor& momentum_buffer,
    const double momentum,
    const double eeta,
    const double eps,
    const double lr,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  float* weight_ptr = weight.data_ptr<float>();
  float* grad_ptr = grad.data_ptr<float>();
  float* momentum_buffer_ptr =
      momentum_buffer.defined() ? momentum_buffer.data_ptr<float>() : nullptr;

  // grad and original weight have same datatype
  using vec_w_t = at::native::Memory::aligned_vector_loop<float, vec_size>;
  using vec_g_t = vec_w_t;

  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_g_t* grad_vec = reinterpret_cast<vec_g_t*>(grad_ptr);

  auto using_momentum = bool(momentum);
  auto using_weight_decay = bool(weight_decay);
  auto weight_decay_value = static_cast<float>(weight_decay);
  auto momentum_value = static_cast<float>(momentum);
  auto negative_lr = static_cast<float>((-1.0) * lr);
  auto eps_value = static_cast<float>(eps);
  auto eeta_value = static_cast<float>(eeta);

  auto w_norm_ptr = w_norm.data_ptr<float>();
  auto g_norm_ptr = g_norm.data_ptr<float>();

  // no master weight
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      // norm op will output one value tensor
      auto w_norm_value = w_norm_ptr[0];
      auto g_norm_value = g_norm_ptr[0];

      // eeta * w_norm / (g_norm + weight_decay * w_norm + eps)
      auto lars_lr = negative_lr;
      if ((w_norm_value > 0.0) && (g_norm_value > 0.0)) {
        lars_lr = negative_lr *
            (eeta * w_norm_value /
             (g_norm_value + weight_decay * w_norm_value + eps_value));
      }

      if (!using_momentum) {
        // vector value
        vec_g_t grad_value = grad_vec[id];
        vec_w_t weight_value = weight_vec[id];
        vec_w_t temp_weight;

#pragma unroll(vec_size)
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          // acc. d_p = p.grad
          auto grad_elm = grad_value[v_index];
          auto temp_weight_value = weight_value[v_index];

          // d_p = d_p.add(p, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elm += temp_weight_value * weight_decay_value;
          }

          // p.add_(d_p, alpha=-group['lr'])
          auto res = temp_weight_value + grad_elm * lars_lr;
          temp_weight[v_index] = res;
        }

        // write back
        weight_vec[id] = temp_weight;

      } else {
        // momentum != 0
        // vector value
        vec_g_t grad_value = grad_vec[id];
        vec_w_t weight_value = weight_vec[id];
        vec_w_t* momentum_buffer_vec =
            reinterpret_cast<vec_w_t*>(momentum_buffer_ptr);

        // momentum buffer vector value
        vec_w_t momentum_buffer_value = momentum_buffer_vec[id];
        vec_w_t temp_weight;
        vec_w_t temp_momentum_buffer;

#pragma unroll(vec_size)
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          // acc. d_p = p.grad
          auto grad_elm = grad_value[v_index];
          auto temp_weight_value = weight_value[v_index];

          // d_p = d_p.add(p, alpha=weight_decay)
          if (using_weight_decay) {
            grad_elm += temp_weight_value * weight_decay_value;
          }

          // 'momentum_buffer' in param_state,
          // param_state[momentum_buffer] has been created
          auto temp_momentum_buffer_value = grad_elm;
          if (momentum_buf_initialized) {
            // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            temp_momentum_buffer_value = momentum_buffer_value[v_index];
            temp_momentum_buffer_value =
                momentum_value * temp_momentum_buffer_value;
            temp_momentum_buffer_value += grad_elm;
          }
          temp_momentum_buffer[v_index] = temp_momentum_buffer_value;

          // d_p = buf
          grad_elm = temp_momentum_buffer_value;

          // p.add_(d_p, alpha=-group['lr'])
          auto res = temp_weight_value + grad_elm * lars_lr;

          // p.data.copy_(p.data)
          temp_weight[v_index] = res;
        }

        // write back
        weight_vec[id] = temp_weight;
        momentum_buffer_vec[id] = temp_momentum_buffer;
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
static void ComputeLARSKernelMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& momentum_buffer,
    Tensor& w_norm,
    Tensor& g_norm,
    const double weight_decay,
    const double momentum,
    const double eeta,
    const double lr,
    const double eps,
    const bool momentum_buf_initialized) {
  TORCH_CHECK(
      master_weight.scalar_type() == at::kFloat,
      "ComputeLARSKernelMasterWeight: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "ComputeLARSKernelMasterWeight: expect grad to be at::BFloat16");
  TORCH_CHECK(
      !momentum_buffer.defined() || momentum_buffer.scalar_type() == at::kFloat,
      "ComputeLARSKernelMasterWeight: expect momentum_buffer to be float32");
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16,
      "ComputeLARSKernelMasterWeight: expect param to be at::kBFloat16");
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto vec_size_grad = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(grad.data_ptr<scalar_t>()));

  auto vec_size_master_weight =
      at::native::Memory::can_vectorize_up_to_loop<float>(
          dpcppGetDeviceIdOfCurrentQueue(),
          reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  auto vec_size_momentum_buffer = vec_size_master_weight;
  // deduce the vector size if need momentum buffer
  if (momentum) {
    vec_size_momentum_buffer =
        at::native::Memory::can_vectorize_up_to_loop<float>(
            dpcppGetDeviceIdOfCurrentQueue(),
            reinterpret_cast<char*>(momentum_buffer.data_ptr<float>()));
  }

  auto vec_size = vec_size_weight;
  if (!master_weight.is_non_overlapping_and_dense() ||
      !grad.is_non_overlapping_and_dense() ||
      !weight.is_non_overlapping_and_dense() ||
      (momentum && !momentum_buffer.is_non_overlapping_and_dense())) {
    vec_size = 1;
  } else {
    vec_size = std::min(
        {vec_size_weight,
         vec_size_grad,
         vec_size_master_weight,
         vec_size_momentum_buffer});
  }

  auto total_element = weight.numel();

  // determine the array size
  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_LARSMW_KERNEL(vec_size)                         \
  {                                                         \
    launch_vec_kernel_LARSMasterWeight<vec_size, scalar_t>( \
        master_weight,                                      \
        weight,                                             \
        grad,                                               \
        w_norm,                                             \
        g_norm,                                             \
        weight_decay,                                       \
        momentum_buf_initialized,                           \
        momentum_buffer,                                    \
        momentum,                                           \
        eeta,                                               \
        eps,                                                \
        lr,                                                 \
        total_element,                                      \
        global_range);                                      \
  }

  switch (vec_size) {
    case 8: {
      VEC_LARSMW_KERNEL(8);
      break;
    }
    case 4: {
      VEC_LARSMW_KERNEL(4);
      break;
    }
    case 2: {
      VEC_LARSMW_KERNEL(2);
      break;
    }
    case 1: {
      VEC_LARSMW_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for launch_vec_kernel_LARSMasterWeight kernel. vec size ",
          vec_size);
  }
#undef VEC_LARSMW_KERNEL
}

static void ComputeLARSKernel(
    Tensor& weight,
    Tensor& grad,
    Tensor& momentum_buffer,
    Tensor& w_norm,
    Tensor& g_norm,
    const double weight_decay,
    const double momentum,
    const double eeta,
    const double lr,
    const double eps,
    const bool momentum_buf_initialized) {
  TORCH_CHECK(
      weight.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect grad to be at::kFloat");
  TORCH_CHECK(
      !momentum_buffer.defined() || momentum_buffer.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect momentum_buffer to be at::kFloat");

  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<float>(
      dpcppGetDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<float>()));

  auto vec_size_grad = at::native::Memory::can_vectorize_up_to_loop<float>(
      dpcppGetDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(grad.data_ptr<float>()));

  auto vec_size_momentum_buffer = vec_size_weight;
  // deduce the vector size if need momentum buffer
  if (momentum) {
    vec_size_momentum_buffer =
        at::native::Memory::can_vectorize_up_to_loop<float>(
            dpcppGetDeviceIdOfCurrentQueue(),
            reinterpret_cast<char*>(momentum_buffer.data_ptr<float>()));
  }

  auto vec_size = vec_size_weight;
  if (!grad.is_non_overlapping_and_dense() ||
      !weight.is_non_overlapping_and_dense() ||
      (momentum && !momentum_buffer.is_non_overlapping_and_dense())) {
    vec_size = 1;
  } else {
    vec_size =
        std::min({vec_size_weight, vec_size_grad, vec_size_momentum_buffer});
  }

  auto total_element = weight.numel();

  // determine the array size
  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_LARS_KERNEL(vec_size)     \
  {                                   \
    launch_vec_kernel_LARS<vec_size>( \
        weight,                       \
        grad,                         \
        w_norm,                       \
        g_norm,                       \
        weight_decay,                 \
        momentum_buf_initialized,     \
        momentum_buffer,              \
        momentum,                     \
        eeta,                         \
        eps,                          \
        lr,                           \
        total_element,                \
        global_range);                \
  }

  switch (vec_size) {
    case 4: {
      VEC_LARS_KERNEL(4);
      break;
    }
    case 2: {
      VEC_LARS_KERNEL(2);
      break;
    }
    case 1: {
      VEC_LARS_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for launch_vec_kernel_LARS kernel. vec size ",
          vec_size);
  }
#undef VEC_LARS_KERNEL
}

} // namespace impl

/**
 * LARS fused update kernel.
 * Support Float, BFloat16 training
 *@param fp32_weight FP32 Parameters or Master Parameter to be updated.
 *@param weight BF16 Parameters of layer support low precision computing.
 *@param grad Grad used to update Parameters.
 *@param momentum_buffer_ momentum to accelerate convergence.
 *@param weight_decay Args for regularization to avoid over-fit.
 *@param momentum Args momentum.
 *@param eeta  Args eeta.
 *@param lr Learning rate.
 *@param eps Args eps.
 */
c10::optional<at::Tensor> lars_fused_step(
    at::Tensor& fp32_weight,
    at::Tensor& weight,
    const at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    double weight_decay,
    double momentum,
    double eeta,
    double lr,
    double eps) {
  RECORD_FUNCTION(
      "lars_fused_step",
      std::vector<c10::IValue>({fp32_weight, weight, grad, momentum_buffer_}));
  const OptionalDeviceGuard device_guard(device_of(fp32_weight));

  // after inference, the model weight in the next training epoch maybe cached
  // block, so to plain now if needed
  fp32_weight = to_plain_if_needed_(fp32_weight);
  at::Tensor grad_processed = to_plain_if_needed(grad);

  at::Tensor w_norm = at::norm(fp32_weight);
  at::Tensor g_norm = at::norm(grad);

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
    grad_processed = grad_processed.contiguous(memory_format);

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "fusion_lars_with_master_weight",
        [&] {
          impl::ComputeLARSKernelMasterWeight<scalar_t>(
              fp32_weight,
              weight,
              grad_processed,
              momentum_buffer,
              w_norm,
              g_norm,
              weight_decay,
              momentum,
              eeta,
              lr,
              eps,
              momentum_buf_initialized);
        });
  } else {
    // normal sgd, no master weight, weight and grad are fp32
    auto memory_format = fp32_weight.suggest_memory_format();
    fp32_weight = fp32_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    grad_processed = grad_processed.contiguous(memory_format);

    // all Tensor are fp32
    impl::ComputeLARSKernel(
        fp32_weight,
        grad_processed,
        momentum_buffer,
        w_norm,
        g_norm,
        weight_decay,
        momentum,
        eeta,
        lr,
        eps,
        momentum_buf_initialized);
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
      "lars_fused_step",
      at::AtenIpexTypeXPU::lars_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace
