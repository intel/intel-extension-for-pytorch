#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
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

template <
    int vec_size,
    typename vec_grad_t,
    typename elem_grad_t,
    typename vec_w_t,
    typename elem_w_t,
    typename scalar_t,
    typename accscalar_t,
    typename vec_mw_t,
    typename elem_mw_t>
void vec_kernel_sgdmw(
    float* master_weight_ptr,
    scalar_t* weight_ptr,
    scalar_t* grad_ptr,
    const double weight_decay,
    const bool momentum_buffer_not_existed,
    float* momentum_buffer_ptr,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr,
    int64_t total_element,
    DPCPP::item<1> item) {
  auto id = item.get_id(0);

  auto remaining = total_element - id * vec_size;

  // cast grad, weight and other memory using vector
  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_grad_t* grad_vec = reinterpret_cast<vec_grad_t*>(grad_ptr);
  vec_mw_t* master_weight_vec = reinterpret_cast<vec_mw_t*>(master_weight_ptr);

  if (!momentum) {
    // for tail
    if (remaining < vec_size) {
      for (auto v_index = 0; v_index < remaining; ++v_index) {
        // kick out tail
        auto linear_idx = id * vec_size + v_index;

        // grad should be fp32 to involve in computation to keep acc.
        // d_p = p.grad.to(p.master_weight.dtype)
        accscalar_t grad_elm = static_cast<accscalar_t>(grad_ptr[linear_idx]);

        float temp_master_weight_value = master_weight_ptr[linear_idx];

        // d_p = d_p.add(p.master_weight, alpha=weight_decay)
        if (weight_decay != 0) {
          grad_elm += temp_master_weight_value * weight_decay;
        }

        // p.master_weight.add_(d_p, alpha=-group['lr'])
        float res = static_cast<float>(
            temp_master_weight_value + grad_elm * float(-1.0) * lr);
        master_weight_ptr[linear_idx] = res;

        // p.data.copy_(p.master_weight.data)
        weight_ptr[linear_idx] = scalar_t(res);
      }
    } else {
      // vector value
      vec_grad_t grad_value = grad_vec[id];
      vec_mw_t master_weight_value = master_weight_vec[id];

      // no tail
      vec_w_t temp_weight;
      vec_mw_t temp_master_weight;

#pragma unroll
      for (auto v_index = 0; v_index < vec_size; ++v_index) {
        // grad should be fp32 to involve in computation to keep acc.
        // d_p = p.grad.to(p.master_weight.dtype)
        auto grad_elm = static_cast<accscalar_t>(
            at::native::Memory::detail::bitwise_cast<scalar_t>(
                grad_value[v_index]));

        auto temp_master_weight_value =
            at::native::Memory::detail::bitwise_cast<float>(
                master_weight_value[v_index]);

        // d_p = d_p.add(p.master_weight, alpha=weight_decay)
        if (weight_decay != 0) {
          grad_elm += temp_master_weight_value * weight_decay;
        }

        // p.master_weight.add_(d_p, alpha=-group['lr'])
        float res = temp_master_weight_value + grad_elm * float(-1.0) * lr;
        temp_master_weight[v_index] =
            at::native::Memory::detail::bitwise_cast<elem_mw_t>(res);

        // p.data.copy_(p.master_weight.data)
        temp_weight[v_index] =
            at::native::Memory::detail::bitwise_cast<elem_w_t>(scalar_t(res));
      }

      // write back
      master_weight_vec[id] = temp_master_weight;
      weight_vec[id] = temp_weight;
    }
  } else {
    // momentum != 0
    // for tail
    if (remaining < vec_size) {
      for (auto v_index = 0; v_index < remaining; ++v_index) {
        // kick out tail
        auto linear_idx = id * vec_size + v_index;

        // grad should be fp32 to involve in computation to keep acc.
        accscalar_t grad_elm = static_cast<accscalar_t>(grad_ptr[linear_idx]);

        float temp_master_weight_value = master_weight_ptr[linear_idx];

        // d_p = d_p.add(p.master_weight, alpha=weight_decay)
        if (weight_decay != 0) {
          grad_elm += temp_master_weight_value * weight_decay;
        }

        // 'momentum_buffer' in param_state, param_state[momentum_buffer] has
        // been created
        float temp_momentum_buffer_value = grad_elm;
        if (!momentum_buffer_not_existed) {
          // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          // here dampening = 1 - dampening
          temp_momentum_buffer_value = momentum_buffer_ptr[linear_idx];
          temp_momentum_buffer_value =
              static_cast<float>(momentum * temp_momentum_buffer_value);
          temp_momentum_buffer_value +=
              static_cast<float>(grad_elm * dampening);
        }
        momentum_buffer_ptr[linear_idx] = temp_momentum_buffer_value;

        // nesterov
        if (nesterov) {
          // d_p = d_p.add(buf, alpha=momentum)
          grad_elm += momentum * temp_momentum_buffer_value;
        } else {
          // d_p = buf
          grad_elm = temp_momentum_buffer_value;
        }

        // p.master_weight.add_(d_p, alpha=-group['lr'])
        float res = static_cast<float>(
            temp_master_weight_value + grad_elm * float(-1.0) * lr);
        master_weight_ptr[linear_idx] = res;

        // p.data.copy_(p.master_weight.data)
        weight_ptr[linear_idx] = scalar_t(res);
      }
    } else {
      // vector value
      vec_grad_t grad_value = grad_vec[id];
      vec_mw_t master_weight_value = master_weight_vec[id];
      vec_mw_t* momentum_buffer_vec =
          reinterpret_cast<vec_mw_t*>(momentum_buffer_ptr);

      // momentum buffer vector value
      vec_mw_t momentum_buffer_value = momentum_buffer_vec[id];

      // no tail
      vec_w_t temp_weight;
      vec_mw_t temp_master_weight;

#pragma unroll
      for (auto v_index = 0; v_index < vec_size; ++v_index) {
        // grad should be fp32 to involve in computation to keep acc.
        // d_p = p.grad.to(p.master_weight.dtype)
        auto grad_elm = static_cast<accscalar_t>(
            at::native::Memory::detail::bitwise_cast<scalar_t>(
                grad_value[v_index]));

        auto temp_master_weight_value =
            at::native::Memory::detail::bitwise_cast<float>(
                master_weight_value[v_index]);

        // d_p = d_p.add(p.master_weight, alpha=weight_decay)
        if (weight_decay != 0) {
          grad_elm += temp_master_weight_value * weight_decay;
        }

        // 'momentum_buffer' in param_state, param_state[momentum_buffer] has
        // been created
        float temp_momentum_buffer_value = grad_elm;
        if (!momentum_buffer_not_existed) {
          // buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          // here dampening = 1 - dampening
          temp_momentum_buffer_value =
              at::native::Memory::detail::bitwise_cast<float>(
                  momentum_buffer_value[v_index]);
          temp_momentum_buffer_value =
              static_cast<float>(momentum * temp_momentum_buffer_value);
          temp_momentum_buffer_value +=
              static_cast<float>(grad_elm * dampening);
        }
        momentum_buffer_vec[id][v_index] =
            at::native::Memory::detail::bitwise_cast<elem_mw_t>(
                temp_momentum_buffer_value);

        // nesterov
        if (nesterov) {
          // d_p = d_p.add(buf, alpha=momentum)
          grad_elm += momentum * temp_momentum_buffer_value;
        } else {
          // d_p = buf
          grad_elm = temp_momentum_buffer_value;
        }

        // p.master_weight.add_(d_p, alpha=-group['lr'])
        float res = static_cast<float>(
            temp_master_weight_value + grad_elm * float(-1.0) * lr);
        temp_master_weight[v_index] =
            at::native::Memory::detail::bitwise_cast<elem_mw_t>(res);

        // p.data.copy_(p.master_weight.data)
        temp_weight[v_index] =
            at::native::Memory::detail::bitwise_cast<elem_w_t>(scalar_t(res));
      }

      // write back
      master_weight_vec[id] = temp_master_weight;
      weight_vec[id] = temp_weight;
    }
  }
}

// use vector R/W to do fusion for SGDMasterWeight optimizer update
template <typename scalar_t>
static void ComputeSGDMasterWeightDecayKernel(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    const double weight_decay,
    const bool momentum_buffer_not_existed,
    Tensor& momentum_buffer,
    const double momentum,
    const double dampening,
    const bool nesterov,
    const double lr) {
  auto& queue = dpcppGetCurrentQueue();

  // In this case, FP32 elements consume more R/W than BF16/FP16 element, so
  // vec_size is chosen same as FP32
  auto vec_size = at::native::Memory::can_vectorize_up_to<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  auto total_element = master_weight.numel();

  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

// launch vector kernel for SGDMasterWeight, code pass according to vector size
#define VEC_SGDMW_KERNEL(vec_size)                                            \
  {                                                                           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                             \
      auto master_weight_ptr = master_weight.data_ptr<float>();               \
      auto weight_ptr = weight.data_ptr<scalar_t>();                          \
      auto grad_ptr = grad.data_ptr<scalar_t>();                              \
      auto momentum_buffer_ptr = momentum_buffer.data_ptr<float>();           \
      using vec_grad_t = typename at::native::Memory::                        \
          aligned_vector<scalar_t, vec_size>::type;                           \
      using elem_grad_t = typename at::native::Memory::                       \
          aligned_vector<scalar_t, vec_size>::element_type;                   \
      using vec_mw_t =                                                        \
          typename at::native::Memory::aligned_vector<float, vec_size>::type; \
      using elem_mw_t = typename at::native::Memory::                         \
          aligned_vector<float, vec_size>::element_type;                      \
      using vec_w_t = vec_grad_t;                                             \
      using elem_w_t = elem_grad_t;                                           \
      using accscalar_t = acc_type<scalar_t>;                                 \
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {                           \
        vec_kernel_sgdmw<                                                     \
            vec_size,                                                         \
            vec_grad_t,                                                       \
            elem_grad_t,                                                      \
            vec_w_t,                                                          \
            elem_w_t,                                                         \
            scalar_t,                                                         \
            accscalar_t,                                                      \
            vec_mw_t,                                                         \
            elem_mw_t>(                                                       \
            master_weight_ptr,                                                \
            weight_ptr,                                                       \
            grad_ptr,                                                         \
            weight_decay,                                                     \
            momentum_buffer_not_existed,                                      \
            momentum_buffer_ptr,                                              \
            momentum,                                                         \
            dampening,                                                        \
            nesterov,                                                         \
            lr,                                                               \
            total_element,                                                    \
            item);                                                            \
      };                                                                      \
      cgh.parallel_for(DPCPP::range<1>(global_range), kfn);                   \
    };                                                                        \
    DPCPP_Q_SUBMIT(queue, cgf);                                               \
  } // namespace impl

  switch (vec_size) {
    case 16: {
      VEC_SGDMW_KERNEL(16);
      break;
    }
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
          "Unexpected vectorization size for SGDMasterWeight kernel. vec size ",
          vec_size);
  }
#undef VEC_SGDMW_KERNEL
} // namespace AtenIpexTypeXPU

} // namespace impl

at::Tensor& fused_SGDMasterWeight(
    at::Tensor& master_weight,
    at::Tensor& weight,
    at::Tensor& grad,
    double weight_decay,
    bool momentum_buffer_not_existed,
    at::Tensor& momentum_buffer,
    double momentum,
    double dampening,
    bool nesterov,
    double lr) {
  RECORD_FUNCTION(
      "fused_SGDMasterWeight",
      std::vector<c10::IValue>(
          {master_weight,
           weight,
           grad,
           weight_decay,
           momentum_buffer_not_existed}));

  auto memory_format = master_weight.suggest_memory_format();
  master_weight = master_weight.contiguous(memory_format);
  weight = weight.contiguous(memory_format);
  grad = grad.contiguous(memory_format);
  momentum_buffer = momentum_buffer.contiguous(memory_format);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "fused_SGDMasterWeight",
      [&] {
        impl::ComputeSGDMasterWeightDecayKernel<scalar_t>(
            master_weight,
            weight,
            grad,
            weight_decay,
            momentum_buffer_not_existed,
            momentum_buffer,
            momentum,
            dampening,
            nesterov,
            lr);
      });
  return master_weight;
}

} // namespace AtenIpexTypeXPU
} // namespace at
