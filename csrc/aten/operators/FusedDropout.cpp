#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <core/Generator.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "Distributions.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include <aten/operators/MemoryAccess.h>

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <int vec_size, typename scalar_t, typename accscalar_t>
void vec_fused_dropout_kernel_impl(
    scalar_t* self_ptr,
    scalar_t* ret_ptr,
    uint8_t* mask_ptr,
    int64_t numel,
    accscalar_t p,
    c10::optional<Generator> gen_) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using mask_vec_t = at::native::Memory::aligned_vector_loop<uint8_t, vec_size>;
  vec_t* self_vec_ptr = reinterpret_cast<vec_t*>(self_ptr);
  vec_t* ret_vec_ptr = reinterpret_cast<vec_t*>(ret_ptr);
  mask_vec_t* mask_vec_ptr = reinterpret_cast<mask_vec_t*>(mask_ptr);
  accscalar_t pinv = accscalar_t(1) / p;

  auto& sycl_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> seeds;
  {
    // See Note [Acquire lock when using random generators]
    // this philox_engine_inputs('1') is aligned with Distribution.cpp,
    // yet they use '((n - 1) / (BLOCK_SIZE * grid.x) + 1) *
    // curand4_engine_calls' in the same place.
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seeds = gen->philox_engine_inputs(1);
  }

  int max_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t num_group =
      (numel + max_group_size * vec_size - 1) / (max_group_size * vec_size);
  sycl::range<1> global_range{num_group * max_group_size};
  sycl::range<1> local_range{max_group_size};
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>{global_range, local_range},
        [=](sycl::nd_item<1> item_id) {
          auto index = item_id.get_global_linear_id();
          RandomState<Philox4_32_10> state(seeds.first, index, seeds.second);
          auto rand = state.uniform<accscalar_t, vec_size>();

          int remaining = numel - index * vec_size;
          if (remaining < vec_size) {
            for (int id = 0; id < remaining; ++id) {
              auto offset = index * vec_size + id;
              auto rv = rand[id] < p;
              ret_ptr[offset] =
                  static_cast<scalar_t>(self_ptr[offset] * rv * pinv);
              mask_ptr[offset] = static_cast<uint8_t>(rv);
            }
          } else {
            vec_t self_value = self_vec_ptr[index];
            mask_vec_t mask_value;
#pragma unroll
            for (int id = 0; id < vec_size; ++id) {
              auto rv = rand[id] < p;
              self_value[id] =
                  static_cast<scalar_t>(self_value[id] * rv * pinv);
              mask_value[id] = static_cast<uint8_t>(rv);
            }
            ret_vec_ptr[index] = self_value;
            mask_vec_ptr[index] = mask_value;
          }
        });
  };

  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void vec_fused_dropout_kernel(
    scalar_t* self_ptr,
    scalar_t* ret_ptr,
    uint8_t* mask_ptr,
    int64_t numel,
    accscalar_t p,
    c10::optional<Generator> gen_) {
  int vec_size_self = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(self_ptr));
  auto vec_size_ret = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(ret_ptr));
  auto vec_size = std::min(vec_size_self, vec_size_ret);

#define VEC_FUSED_DROPOUT_KERNEL_IMPL(vec_size)                     \
  {                                                                 \
    vec_fused_dropout_kernel_impl<vec_size, scalar_t, accscalar_t>( \
        self_ptr, ret_ptr, mask_ptr, numel, p, gen_);               \
  }

  switch (vec_size) {
    case 8: {
      VEC_FUSED_DROPOUT_KERNEL_IMPL(8);
      break;
    }
    case 4: {
      VEC_FUSED_DROPOUT_KERNEL_IMPL(4);
      break;
    }
    case 2: {
      VEC_FUSED_DROPOUT_KERNEL_IMPL(2);
      break;
    }
    case 1: {
      VEC_FUSED_DROPOUT_KERNEL_IMPL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for FusedDropout. vec size ",
          vec_size);
  }
#undef VEC_FUSED_DROPOUT_KERNEL_IMPL
}

template <typename scalar_t, typename accscalar_t>
void masked_scale_kernel(
    at::Tensor& ret,
    const Tensor& self,
    const Tensor mask,
    accscalar_t scale) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto self_info = getTensorInfo<scalar_t, uint64_t>(self);
  auto mask_info = getTensorInfo<uint8_t, uint64_t>(mask);
  self_info.collapseDims();
  mask_info.collapseDims();

  int64_t numel = self.numel();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_ptr = self.data_ptr<scalar_t>();
    auto ret_ptr = ret.data_ptr<scalar_t>();
    auto mask_ptr = mask.data_ptr<uint8_t>();

    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      auto index = item_id.get_linear_id();
      auto self_offset =
          IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
      auto mask_offset =
          IndexToOffset<uint8_t, uint64_t>::get(index, mask_info);
      ret_ptr[index] = self_ptr[self_offset] *
          static_cast<uint8_t>(mask_ptr[mask_offset]) * scale;
    };
    cgh.parallel_for(sycl::range<1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <int vec_size, typename scalar_t, typename accscalar_t>
void vec_masked_scale_kernel_impl(
    scalar_t* ret_ptr,
    scalar_t* self_ptr,
    uint8_t* mask_ptr,
    int64_t numel,
    accscalar_t scale) {
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using mask_vec_t = at::native::Memory::aligned_vector_loop<uint8_t, vec_size>;
  vec_t* self_vec = reinterpret_cast<vec_t*>(self_ptr);
  vec_t* ret_vec = reinterpret_cast<vec_t*>(ret_ptr);
  mask_vec_t* mask_vec = reinterpret_cast<mask_vec_t*>(mask_ptr);

  auto& sycl_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t num_group =
      (numel + max_group_size * vec_size - 1) / (max_group_size * vec_size);
  sycl::range<1> global_range{num_group * max_group_size};
  sycl::range<1> local_range{max_group_size};
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>{global_range, local_range},
        [=](sycl::nd_item<1> item_id) {
          auto index = item_id.get_global_linear_id();
          int remaining = numel - vec_size * index;

          if (remaining < vec_size) {
            for (int id = 0; id < remaining; ++id) {
              auto offset = index * vec_size + id;
              auto self_val = self_ptr[offset];
              auto mask_val = mask_ptr[offset];
              scalar_t ret_val =
                  static_cast<scalar_t>(self_val * mask_val * scale);
              ret_ptr[offset] = ret_val;
            }
          } else {
            auto self_value = self_vec[index];
            auto mask_value = mask_vec[index];
            vec_t ret_value;
#pragma unroll
            for (int id = 0; id < vec_size; ++id) {
              ret_value[id] = static_cast<scalar_t>(
                  self_value[id] * mask_value[id] * scale);
            }
            ret_vec[index] = ret_value;
          }
        });
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void vec_masked_scale_kernel(
    scalar_t* ret_ptr,
    scalar_t* self_ptr,
    uint8_t* mask_ptr,
    int64_t numel,
    accscalar_t scale) {
  auto vec_size_self = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(self_ptr));
  auto vec_size_ret = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(ret_ptr));
  auto vec_size = std::min(vec_size_self, vec_size_ret);

#define VEC_MASKED_SCALE_KERNEL_IMPL(vec_size)                     \
  {                                                                \
    vec_masked_scale_kernel_impl<vec_size, scalar_t, accscalar_t>( \
        ret_ptr, self_ptr, mask_ptr, numel, scale);                \
  }

  switch (vec_size) {
    case 8: {
      VEC_MASKED_SCALE_KERNEL_IMPL(8);
      break;
    }
    case 4: {
      VEC_MASKED_SCALE_KERNEL_IMPL(4);
      break;
    }
    case 2: {
      VEC_MASKED_SCALE_KERNEL_IMPL(2);
      break;
    }
    case 1: {
      VEC_MASKED_SCALE_KERNEL_IMPL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for MaskedScale. vec size ",
          vec_size);
  }
#undef VEC_MASKED_SCALE_KERNEL_IMPL
}

} // namespace impl

std::tuple<Tensor, Tensor> _fused_dropout(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen_) {
  Tensor input = self.contiguous();
  Tensor ret = at::empty_like(self, self.suggest_memory_format());
  Tensor mask = at::empty(
      self.sizes(), self.options().dtype(kByte), self.suggest_memory_format());

  // TODO: Should add path for not satisfy canUse32BitIndexMath
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_dropout_kernel",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        accscalar_t pa = (accscalar_t)(p);
        impl::vec_fused_dropout_kernel<scalar_t, accscalar_t>(
            input.data_ptr<scalar_t>(),
            ret.data_ptr<scalar_t>(),
            mask.data_ptr<uint8_t>(),
            self.numel(),
            pa,
            gen_);
      });

  return std::tuple<Tensor, Tensor>(ret, mask);
}

Tensor _masked_scale(const Tensor& self, const Tensor& mask, double scale) {
  auto input = self.contiguous();
  auto _mask = mask.contiguous();
  Tensor ret = at::empty_like(self, self.suggest_memory_format());

  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "masked_scale",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        accscalar_t pa = (accscalar_t)(scale);
        impl::vec_masked_scale_kernel<scalar_t, accscalar_t>(
            ret.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            _mask.data_ptr<uint8_t>(),
            self.numel(),
            pa);
      });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
