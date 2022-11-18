#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <torch/library.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <int vec_size, typename scalar_t>
void launch_vec_kernel_sgdMW(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    const double weight_decay,
    const bool momentum_buf_initialized,
    Tensor& momentum_buffer,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  float* master_weight_ptr = master_weight.data_ptr<float>();
  scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
  scalar_t* grad_ptr = grad.data_ptr<scalar_t>();
  float* momentum_buffer_ptr =
      momentum_buffer.defined() ? momentum_buffer.data_ptr<float>() : nullptr;

  using vec_mw_t = at::native::Memory::aligned_vector_loop<float, vec_size>;
  using vec_w_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using vec_g_t = vec_w_t;

  vec_mw_t* master_weight_vec = reinterpret_cast<vec_mw_t*>(master_weight_ptr);
  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_g_t* grad_vec = reinterpret_cast<vec_g_t*>(grad_ptr);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      auto remaining = total_element - id * vec_size;

      if (!momentum) {
        if (remaining < vec_size) {
          for (auto v_index = 0; v_index < remaining; ++v_index) {
            // kick out tail
            auto linear_idx = id * vec_size + v_index;

            // grad should be fp32 to involve in computation to keep acc.
            // d_p = p.grad.to(p.master_weight.dtype)
            auto grad_elm = static_cast<float>(grad_ptr[linear_idx]);

            auto temp_master_weight_value = master_weight_ptr[linear_idx];

            // d_p = d_p.add(p.master_weight, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_master_weight_value * weight_decay;
            }

            // p.master_weight.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<float>(
                temp_master_weight_value + grad_elm * (-1.0) * lr);
            master_weight_ptr[linear_idx] = res;

            // p.data.copy_(p.master_weight.data)
            weight_ptr[linear_idx] = static_cast<scalar_t>(res);
          }
        } else {
          // vector value
          vec_g_t grad_value = grad_vec[id];
          vec_mw_t master_weight_value = master_weight_vec[id];

          vec_w_t temp_weight;
          vec_mw_t temp_master_weight;

#pragma unroll
          for (auto v_index = 0; v_index < vec_size; ++v_index) {
            // grad should be fp32 to involve in computation to keep acc.
            // d_p = p.grad.to(p.master_weight.dtype)
            auto grad_elm = static_cast<float>(grad_value[v_index]);

            auto temp_master_weight_value = master_weight_value[v_index];

            // d_p = d_p.add(p.master_weight, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_master_weight_value * weight_decay;
            }

            // p.master_weight.add_(d_p, alpha=-group['lr'])
            auto res = temp_master_weight_value + grad_elm * (-1.0) * lr;
            temp_master_weight[v_index] = static_cast<float>(res);

            // p.data.copy_(p.master_weight.data)
            temp_weight[v_index] = static_cast<scalar_t>(res);
          }

          // write back
          master_weight_vec[id] = temp_master_weight;
          weight_vec[id] = temp_weight;
        }
      } else {
        // momentum != 0
        if (remaining < vec_size) {
          for (auto v_index = 0; v_index < remaining; ++v_index) {
            // kick out tail
            auto linear_idx = id * vec_size + v_index;

            // grad should be fp32 to involve in computation to keep acc.
            auto grad_elm = static_cast<float>(grad_ptr[linear_idx]);

            auto temp_master_weight_value = master_weight_ptr[linear_idx];

            // d_p = d_p.add(p.master_weight, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_master_weight_value * weight_decay;
            }

            // 'momentum_buffer' in param_state,
            // param_state[momentum_buffer] has been created
            auto temp_momentum_buffer_value = grad_elm;
            if (momentum_buf_initialized) {
              // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
              temp_momentum_buffer_value = momentum_buffer_ptr[linear_idx];
              temp_momentum_buffer_value =
                  momentum * temp_momentum_buffer_value;
              temp_momentum_buffer_value += grad_elm * (1.0 - dampening);
            }
            momentum_buffer_ptr[linear_idx] =
                static_cast<float>(temp_momentum_buffer_value);

            // nesterov
            if (nesterov) {
              // d_p = d_p.add(buf, alpha=momentum)
              grad_elm += momentum * temp_momentum_buffer_value;
            } else {
              // d_p = buf
              grad_elm = temp_momentum_buffer_value;
            }

            // p.master_weight.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<float>(
                temp_master_weight_value + grad_elm * (-1.0) * lr);
            master_weight_ptr[linear_idx] = res;

            // p.data.copy_(p.master_weight.data)
            weight_ptr[linear_idx] = static_cast<scalar_t>(res);
          }
        } else {
          // vector value
          vec_g_t grad_value = grad_vec[id];
          vec_mw_t master_weight_value = master_weight_vec[id];
          vec_mw_t* momentum_buffer_vec =
              reinterpret_cast<vec_mw_t*>(momentum_buffer_ptr);

          // momentum buffer vector value
          vec_mw_t momentum_buffer_value = momentum_buffer_vec[id];
          vec_w_t temp_weight;
          vec_mw_t temp_master_weight;
          vec_mw_t temp_momentum_buffer;

#pragma unroll
          for (auto v_index = 0; v_index < vec_size; ++v_index) {
            // grad should be fp32 to involve in computation to keep acc.
            // d_p = p.grad.to(p.master_weight.dtype)
            auto grad_elm = static_cast<float>(grad_value[v_index]);

            auto temp_master_weight_value = master_weight_value[v_index];

            // d_p = d_p.add(p.master_weight, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_master_weight_value * weight_decay;
            }

            // 'momentum_buffer' in param_state,
            // param_state[momentum_buffer] has been created
            auto temp_momentum_buffer_value = grad_elm;
            if (momentum_buf_initialized) {
              // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
              temp_momentum_buffer_value = momentum_buffer_value[v_index];
              temp_momentum_buffer_value =
                  momentum * temp_momentum_buffer_value;
              temp_momentum_buffer_value += grad_elm * (1.0 - dampening);
            }
            temp_momentum_buffer[v_index] =
                static_cast<float>(temp_momentum_buffer_value);

            // nesterov
            if (nesterov) {
              // d_p = d_p.add(buf, alpha=momentum)
              grad_elm += momentum * temp_momentum_buffer_value;
            } else {
              // d_p = buf
              grad_elm = temp_momentum_buffer_value;
            }

            // p.master_weight.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<float>(
                temp_master_weight_value + grad_elm * (-1.0) * lr);
            temp_master_weight[v_index] = res;

            // p.data.copy_(p.master_weight.data)
            temp_weight[v_index] = static_cast<scalar_t>(res);
          }

          // write back
          master_weight_vec[id] = temp_master_weight;
          weight_vec[id] = temp_weight;
          momentum_buffer_vec[id] = temp_momentum_buffer;
        }
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <int vec_size, typename scalar_t>
void launch_vec_kernel_sgd(
    Tensor& weight,
    Tensor& grad,
    const double weight_decay,
    const bool momentum_buf_initialized,
    Tensor& momentum_buffer,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
  scalar_t* grad_ptr = grad.data_ptr<scalar_t>();
  scalar_t* momentum_buffer_ptr = momentum_buffer.defined()
      ? momentum_buffer.data_ptr<scalar_t>()
      : nullptr;

  // grad and original weight have same datatype
  using vec_w_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using vec_g_t = vec_w_t;

  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_g_t* grad_vec = reinterpret_cast<vec_g_t*>(grad_ptr);

  // no master weight
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      auto remaining = total_element - id * vec_size;
      if (!momentum) {
        if (remaining < vec_size) {
          for (auto v_index = 0; v_index < remaining; ++v_index) {
            // kick out tail
            auto linear_idx = id * vec_size + v_index;

            auto temp_weight_value = weight_ptr[linear_idx];
            auto grad_elm = grad_ptr[linear_idx];

            // d_p = d_p.add(p.master_weight, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_weight_value * weight_decay;
            }

            // p.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<scalar_t>(
                temp_weight_value + grad_elm * (-1.0) * lr);
            weight_ptr[linear_idx] = res;
          }
        } else {
          // vector value
          vec_g_t grad_value = grad_vec[id];
          vec_w_t weight_value = weight_vec[id];
          vec_w_t temp_weight;

#pragma unroll
          for (auto v_index = 0; v_index < vec_size; ++v_index) {
            // acc. d_p = p.grad
            auto grad_elm = grad_value[v_index];
            auto temp_weight_value = weight_value[v_index];

            // d_p = d_p.add(p, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_weight_value * weight_decay;
            }

            // p.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<scalar_t>(
                temp_weight_value + grad_elm * (-1.0) * lr);
            temp_weight[v_index] = res;
          }

          // write back
          weight_vec[id] = temp_weight;
        }
      } else {
        // momentum != 0
        if (remaining < vec_size) {
          for (auto v_index = 0; v_index < remaining; ++v_index) {
            // kick out tail
            auto linear_idx = id * vec_size + v_index;

            auto grad_elm = grad_ptr[linear_idx];
            auto temp_weight_value = weight_ptr[linear_idx];

            // d_p = d_p.add(p, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_weight_value * weight_decay;
            }

            // 'momentum_buffer' in param_state,
            // param_state[momentum_buffer] has been created
            auto temp_momentum_buffer_value = grad_elm;
            if (momentum_buf_initialized) {
              // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
              temp_momentum_buffer_value = momentum_buffer_ptr[linear_idx];
              temp_momentum_buffer_value =
                  momentum * temp_momentum_buffer_value;
              temp_momentum_buffer_value += grad_elm * (1.0 - dampening);
            }
            momentum_buffer_ptr[linear_idx] =
                static_cast<scalar_t>(temp_momentum_buffer_value);

            // nesterov
            if (nesterov) {
              // d_p = d_p.add(buf, alpha=momentum)
              grad_elm += momentum * temp_momentum_buffer_value;
            } else {
              // d_p = buf
              grad_elm = temp_momentum_buffer_value;
            }

            // p.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<scalar_t>(
                temp_weight_value + grad_elm * (-1.0) * lr);

            // write back
            weight_ptr[linear_idx] = res;
          }
        } else {
          // vector value
          vec_g_t grad_value = grad_vec[id];
          vec_w_t weight_value = weight_vec[id];
          vec_w_t* momentum_buffer_vec =
              reinterpret_cast<vec_w_t*>(momentum_buffer_ptr);

          // momentum buffer vector value
          vec_w_t momentum_buffer_value = momentum_buffer_vec[id];
          vec_w_t temp_weight;
          vec_w_t temp_momentum_buffer;

#pragma unroll
          for (auto v_index = 0; v_index < vec_size; ++v_index) {
            // acc. d_p = p.grad
            auto grad_elm = grad_value[v_index];
            auto temp_weight_value = weight_value[v_index];

            // d_p = d_p.add(p, alpha=weight_decay)
            if (weight_decay != 0) {
              grad_elm += temp_weight_value * weight_decay;
            }

            // 'momentum_buffer' in param_state,
            // param_state[momentum_buffer] has been created
            auto temp_momentum_buffer_value = grad_elm;
            if (momentum_buf_initialized) {
              // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
              temp_momentum_buffer_value = momentum_buffer_value[v_index];
              temp_momentum_buffer_value =
                  momentum * temp_momentum_buffer_value;
              temp_momentum_buffer_value += grad_elm * (1.0 - dampening);
            }
            temp_momentum_buffer[v_index] =
                static_cast<scalar_t>(temp_momentum_buffer_value);

            // nesterov
            if (nesterov) {
              // d_p = d_p.add(buf, alpha=momentum)
              grad_elm += momentum * temp_momentum_buffer_value;
            } else {
              // d_p = buf
              grad_elm = temp_momentum_buffer_value;
            }

            // p.add_(d_p, alpha=-group['lr'])
            auto res = static_cast<scalar_t>(
                temp_weight_value + grad_elm * (-1.0) * lr);

            // p.data.copy_(p.data)
            temp_weight[v_index] = res;
          }

          // write back
          weight_vec[id] = temp_weight;
          momentum_buffer_vec[id] = temp_momentum_buffer;
        }
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

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
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto vec_size_master_weight =
      at::native::Memory::can_vectorize_up_to_loop<float>(
          getDeviceIdOfCurrentQueue(),
          reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  auto vec_size = std::min(vec_size_master_weight, vec_size_weight);

  auto total_element = weight.numel();

  // determine the array size
  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_SGDMW_KERNEL(vec_size)               \
  {                                              \
    launch_vec_kernel_sgdMW<vec_size, scalar_t>( \
        master_weight,                           \
        weight,                                  \
        grad,                                    \
        weight_decay,                            \
        momentum_buf_initialized,                \
        momentum_buffer,                         \
        momentum,                                \
        dampening,                               \
        nesterov,                                \
        lr,                                      \
        total_element,                           \
        global_range);                           \
  }

  switch (vec_size) {
    case 8: {
      VEC_SGDMW_KERNEL(8);
      break;
    }
    case 4: {
      VEC_SGDMW_KERNEL(4);
      break;
    }
    case 2: {
      VEC_SGDMW_KERNEL(2);
      break;
    }
    case 1: {
      VEC_SGDMW_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for launch_vec_kernel_sgdMW kernel. vec size ",
          vec_size);
  }
#undef VEC_SGDMW_KERNEL
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
      weight.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect grad to be at::kFloat");
  TORCH_CHECK(
      !momentum_buffer.defined() || momentum_buffer.scalar_type() == at::kFloat,
      "ComputeSGDKernel: expect momentum_buffer to be at::kFloat");

  auto& queue = dpcppGetCurrentQueue();

  auto vec_size = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto total_element = weight.numel();

  // determine the array size
  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_SGD_KERNEL(vec_size)               \
  {                                            \
    launch_vec_kernel_sgd<vec_size, scalar_t>( \
        weight,                                \
        grad,                                  \
        weight_decay,                          \
        momentum_buf_initialized,              \
        momentum_buffer,                       \
        momentum,                              \
        dampening,                             \
        nesterov,                              \
        lr,                                    \
        total_element,                         \
        global_range);                         \
  }

  switch (vec_size) {
    case 4: {
      VEC_SGD_KERNEL(4);
      break;
    }
    case 2: {
      VEC_SGD_KERNEL(2);
      break;
    }
    case 1: {
      VEC_SGD_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for launch_vec_kernel_sgd kernel. vec size ",
          vec_size);
  }
#undef VEC_SGD_KERNEL
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
    at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    at::Tensor& weight,
    const double momentum,
    const double lr,
    const double weight_decay,
    const double dampening,
    const bool nesterov) {
  RECORD_FUNCTION(
      "sgd_fused_step",
      std::vector<c10::IValue>({fp32_weight, grad, momentum_buffer_, weight}));

  at::Tensor momentum_buffer;
  bool momentum_buf_initialized;
  if (momentum != 0) {
    if (!momentum_buffer_.has_value()) {
      momentum_buffer = at::empty_like(fp32_weight);
      momentum_buf_initialized = false;
    } else {
      momentum_buffer = momentum_buffer_.value().contiguous();
      momentum_buf_initialized = true;
    }
  }

  // master weight mode, fp32_weight contains fp32 master weight, weight is
  // bf16 weight, grad is bf16
  if (weight.numel()) {
    auto memory_format = weight.suggest_memory_format();
    fp32_weight = fp32_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    weight = weight.contiguous(memory_format);
    grad = grad.contiguous(memory_format);

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "fusion_sgd_with_master_weight",
        [&] {
          impl::ComputeSGDKernelMasterWeight<scalar_t>(
              fp32_weight,
              grad,
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
    // normal sgd, no master weight, weight and grad are fp32
    auto memory_format = fp32_weight.suggest_memory_format();
    fp32_weight = fp32_weight.contiguous(memory_format);

    if (momentum_buffer.numel()) {
      momentum_buffer = momentum_buffer.contiguous(memory_format);
    }

    grad = grad.contiguous(memory_format);

    // all Tensor are fp32
    IPEX_DISPATCH_FLOATING_TYPES(
        fp32_weight.scalar_type(), "fusion_sgd_no_master_weight", [&] {
          impl::ComputeSGDKernel<scalar_t>(
              fp32_weight,
              grad,
              momentum_buffer,
              weight_decay,
              momentum,
              dampening,
              nesterov,
              lr,
              momentum_buf_initialized);
        });
  }

  if (momentum == 0) {
    return c10::nullopt;
  } else {
    return momentum_buffer;
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "sgd_fused_step(Tensor fp32_weight, Tensor grad, Tensor? momentum_buffer_, "
      "Tensor weight, float momentum, float lr, float weight_decay, "
      "float dampening, bool nesterov) -> Tensor?");
  m.impl(
      "sgd_fused_step",
      c10::DispatchKey::XPU,
      at::AtenIpexTypeXPU::sgd_fused_step);
}
} // namespace
