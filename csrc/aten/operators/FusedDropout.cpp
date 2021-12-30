#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <core/Generator.h>
#include <runtime/Utils.h>
#include "Distributions.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <utils/oneMKLUtils.h>
#endif

#include <aten/operators/MemoryAccess.h>

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {
// TODO: use operator DistributionBernoulli to replace this function when
// performance is ready.
template <typename scalar_t, typename accscalar_t>
void bernoulli_distr_kernel(
    const Tensor& self,
    Tensor& rand,
    accscalar_t p,
    std::pair<uint64_t, uint64_t> seeds) {
  RECORD_FUNCTION("bernoulliDistr", {});
  auto& sycl_queue = dpcppGetCurrentQueue();

  // generate bernoulli distribution
  // oneMKL generator only surpport int32_t and uint32_t datatype
  int32_t* rand_ptr = rand.data_ptr<int32_t>();
  int64_t numel = self.numel();
  std::initializer_list<std::uint64_t> seed = {seeds.first, 0, seeds.second};
  float val = static_cast<float>(accscalar_t(1) - p);

#ifdef USE_ONEMKL
  oneapi::mkl::rng::philox4x32x10 engine(sycl_queue, seed);
  oneapi::mkl::rng::bernoulli<int32_t> distr(val);
  auto e = oneapi::mkl::rng::generate(distr, engine, numel, rand_ptr);
  dpcpp_log("dpcpp_kernel", e);
  DPCPP_E_SYNC_FOR_DEBUG(e);
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

Tensor bernoulliDistr_impl(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen_) {
  at::Tensor rand = at::empty(
      self.sizes(), self.options().dtype(kInt), self.suggest_memory_format());
  int64_t nelem = self.numel();
  // empty tensors should not get here, but just in case, avoid FPE
  if (nelem == 0)
    return rand;

  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }

  // TODO: Should add path for not satisfy canUse32BitIndexMath
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bernoulli_distr_kernel",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        accscalar_t pa = (accscalar_t)(p);
        bernoulli_distr_kernel<scalar_t, accscalar_t>(
            self, rand, p, rng_engine_inputs);
      });
  return rand;
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename rscalar_t>
void vec_fused_dropout_kernel_impl(
    const Tensor& self,
    const Tensor& rand,
    Tensor& ret,
    Tensor& mask,
    accscalar_t p) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  int64_t numel = self.numel();
  int64_t global_range = (numel + vec_size - 1) / vec_size;

  auto self_ptr = self.data_ptr<scalar_t>();
  auto ret_ptr = ret.data_ptr<scalar_t>();
  using vec_t =
      typename at::native::Memory::aligned_vector<scalar_t, vec_size>::type;
  using elem_t = typename at::native::Memory::
      aligned_vector<scalar_t, vec_size>::element_type;
  vec_t* self_vec = reinterpret_cast<vec_t*>(self_ptr);
  vec_t* ret_vec = reinterpret_cast<vec_t*>(ret_ptr);

  rscalar_t* rand_ptr = rand.data_ptr<rscalar_t>();
  using rscalar_vec_t =
      typename at::native::Memory::aligned_vector<rscalar_t, vec_size>::type;
  rscalar_vec_t* rand_vec = reinterpret_cast<rscalar_vec_t*>(rand_ptr);

  auto mask_ptr = mask.data_ptr<uint8_t>();
  using mask_vec_t =
      typename at::native::Memory::aligned_vector<uint8_t, vec_size>::type;
  using mask_elem_t = typename at::native::Memory::
      aligned_vector<uint8_t, vec_size>::element_type;
  mask_vec_t* mask_vec = reinterpret_cast<mask_vec_t*>(mask_ptr);

  accscalar_t pinv = accscalar_t(1) / p;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto index = item_id.get_linear_id();
      int remaining = numel - index * vec_size;

      if (remaining < vec_size) {
        for (int id = 0; id < remaining; ++id) {
          auto offset = index * vec_size + id;
          auto self_val = self_ptr[offset];
          auto rand_val = rand_ptr[offset];
          auto rv = accscalar_t(rand_val) < p;
          ret_ptr[offset] = static_cast<scalar_t>(self_val * rv * pinv);
          mask_ptr[offset] = static_cast<uint8_t>(rv);
        }
      } else {
        vec_t self_value = self_vec[index];
        rscalar_vec_t rand_value = rand_vec[index];
        vec_t ret_value;
        mask_vec_t mask_value;
#pragma unroll
        for (int id = 0; id < vec_size; ++id) {
          auto self_val = at::native::Memory::detail::bitwise_cast<scalar_t>(
              self_value[id]);
          auto rand_val = at::native::Memory::detail::bitwise_cast<rscalar_t>(
              rand_value[id]);
          auto rv = accscalar_t(rand_val) < p;
          auto ret_val = static_cast<scalar_t>(self_val * rv * pinv);
          auto mask_val = static_cast<uint8_t>(rv);
          ret_value[id] =
              at::native::Memory::detail::bitwise_cast<elem_t>(ret_val);
          mask_value[id] =
              at::native::Memory::detail::bitwise_cast<mask_elem_t>(mask_val);
        }
        ret_vec[index] = ret_value;
        mask_vec[index] = mask_value;
      }
    };
    cgh.parallel_for(DPCPP::range<1>(global_range), kfn);
  };

  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <typename scalar_t, typename accscalar_t, typename rscalar_t>
void vec_fused_dropout_kernel(
    const Tensor& self,
    const Tensor& rand,
    Tensor& ret,
    Tensor& mask,
    accscalar_t p) {
  // Two tensors (self and ret) are in scalar_t
  // Only one tensor (rand) is in rscalar_t
  // Only one tensor (mask) is in uint8_t
  // so scalar_t is used to query the vec_size
  auto self_ptr = self.data_ptr<scalar_t>();
  auto ret_ptr = ret.data_ptr<scalar_t>();
  auto vec_size_self = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(self_ptr));
  auto vec_size_ret = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(ret_ptr));
  // set the min value of self and rand as the final vec_size
  auto vec_size = std::min(vec_size_self, vec_size_ret);

#define VEC_FUSED_DROPOUT_KERNEL_IMPL(vec_size)                                \
  {                                                                            \
    vec_fused_dropout_kernel_impl<vec_size, scalar_t, accscalar_t, rscalar_t>( \
        self, rand, ret, mask, p);                                             \
  }

  // TODO: we need to consider how to reduce the kernel size
  switch (vec_size) {
    case 16: {
      VEC_FUSED_DROPOUT_KERNEL_IMPL(16);
      break;
    }
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

template <typename scalar_t, typename accscalar_t, typename rscalar_t>
void fused_dropout_kernel(
    const Tensor& self,
    const Tensor& rand,
    Tensor& ret,
    Tensor& mask,
    accscalar_t p) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto self_info = getTensorInfo<scalar_t, uint64_t>(self);
  self_info.collapseDims();
  rscalar_t* rand_ptr = rand.data_ptr<rscalar_t>();
  int64_t numel = self.numel();

  accscalar_t pinv = accscalar_t(1) / p;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_ptr = self.data_ptr<scalar_t>();
    auto ret_ptr = ret.data_ptr<scalar_t>();
    auto mask_ptr = mask.data_ptr<uint8_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto index = item_id.get_linear_id();
      auto self_offset =
          IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
      auto rv = accscalar_t(rand_ptr[index]) < p;
      if (index < numel) {
        ret_ptr[index] = self_ptr[self_offset] * rv * pinv;
        mask_ptr[index] = (uint8_t)rv;
      }
    };
    cgh.parallel_for(DPCPP::range<1>(numel), kfn);
  };

  DPCPP_Q_SUBMIT(sycl_queue, cgf);
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

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto index = item_id.get_linear_id();
      auto self_offset =
          IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
      auto mask_offset =
          IndexToOffset<uint8_t, uint64_t>::get(index, mask_info);
      ret_ptr[index] = self_ptr[self_offset] *
          static_cast<uint8_t>(mask_ptr[mask_offset]) * scale;
    };
    cgh.parallel_for(DPCPP::range<1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <int vec_size, typename scalar_t, typename accscalar_t>
void vec_masked_scale_kernel_impl(
    at::Tensor& ret,
    const Tensor& self,
    const Tensor mask,
    accscalar_t scale) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  int64_t numel = self.numel();
  int64_t global_range = (numel + vec_size - 1) / vec_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_ptr = self.data_ptr<scalar_t>();
    auto ret_ptr = ret.data_ptr<scalar_t>();
    using vec_t =
        typename at::native::Memory::aligned_vector<scalar_t, vec_size>::type;
    using elem_t = typename at::native::Memory::
        aligned_vector<scalar_t, vec_size>::element_type;
    vec_t* self_vec = reinterpret_cast<vec_t*>(self_ptr);
    vec_t* ret_vec = reinterpret_cast<vec_t*>(ret_ptr);

    auto mask_ptr = mask.data_ptr<uint8_t>();
    using mask_vec_t =
        typename at::native::Memory::aligned_vector<uint8_t, vec_size>::type;
    mask_vec_t* mask_vec = reinterpret_cast<mask_vec_t*>(mask_ptr);

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto index = item_id.get_linear_id();
      int remaining = numel - vec_size * index;

      if (remaining < vec_size) {
        for (int id = 0; id < remaining; ++id) {
          auto offset = index * vec_size + id;
          auto self_val = self_ptr[offset];
          auto mask_val = mask_ptr[offset];
          scalar_t ret_val = static_cast<scalar_t>(self_val * mask_val * scale);
          ret_ptr[offset] = ret_val;
        }
      } else {
        auto self_value = self_vec[index];
        auto mask_value = mask_vec[index];
        vec_t ret_value;
#pragma unroll
        for (int id = 0; id < vec_size; ++id) {
          auto self_val = at::native::Memory::detail::bitwise_cast<scalar_t>(
              self_value[id]);
          auto mask_val =
              at::native::Memory::detail::bitwise_cast<uint8_t>(mask_value[id]);
          scalar_t ret_val = static_cast<scalar_t>(self_val * mask_val * scale);
          ret_value[id] =
              at::native::Memory::detail::bitwise_cast<elem_t>(ret_val);
        }
        ret_vec[index] = ret_value;
      }
    };
    cgh.parallel_for(DPCPP::range<1>(global_range), kfn);
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void vec_masked_scale_kernel(
    at::Tensor& ret,
    const Tensor& self,
    const Tensor mask,
    accscalar_t scale) {
  // Two tensors (self and ret) are in scalar_t
  // Only one tensor (mask) is in uint8_t
  // So the scalar_t is used to query the vec_size
  auto self_ptr = self.data_ptr<scalar_t>();
  auto ret_ptr = ret.data_ptr<scalar_t>();
  auto vec_size_self = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(self_ptr));
  auto vec_size_ret = at::native::Memory::can_vectorize_up_to<scalar_t>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(ret_ptr));
  // set the min value of self and ret as the final vec_size
  auto vec_size = std::min(vec_size_self, vec_size_ret);

#define VEC_MASKED_SCALE_KERNEL_IMPL(vec_size)                     \
  {                                                                \
    vec_masked_scale_kernel_impl<vec_size, scalar_t, accscalar_t>( \
        ret, self, mask, scale);                                   \
  }

  // TODO: we need to consider how to reduce the kernel size
  switch (vec_size) {
    case 16: {
      VEC_MASKED_SCALE_KERNEL_IMPL(16);
      break;
    }
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
#ifdef USE_ONEMKL
        Tensor rand = impl::bernoulliDistr_impl(self, p, gen_);
        if (self.is_contiguous()) {
          impl::vec_fused_dropout_kernel<scalar_t, accscalar_t, int32_t>(
              self, rand, ret, mask, pa);
        } else {
          impl::fused_dropout_kernel<scalar_t, accscalar_t, int32_t>(
              self, rand, ret, mask, pa);
        }
#else
        at::Tensor rand = at::empty_like(self, self.suggest_memory_format());
        rand.bernoulli_(1 - p);
        rand.div_(1 - p);
        if (self.is_contiguous()) {
          impl::vec_fused_dropout_kernel<scalar_t, accscalar_t, scalar_t>(
              self, rand, ret, mask, pa);
        } else {
          impl::fused_dropout_kernel<scalar_t, accscalar_t, scalar_t>(
              self, rand, ret, mask, pa);
        }
#endif
      });

  return std::tuple<Tensor, Tensor>(ret, mask);
}

Tensor _masked_scale(const Tensor& self, const Tensor& mask, double scale) {
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
        if (self.is_contiguous() && mask.is_contiguous()) {
          impl::vec_masked_scale_kernel<scalar_t, accscalar_t>(
              ret, self, mask, pa);
        } else {
          impl::masked_scale_kernel<scalar_t, accscalar_t>(ret, self, mask, pa);
        }
      });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
