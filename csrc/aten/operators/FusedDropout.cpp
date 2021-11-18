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
  auto self_info = getTensorInfo<scalar_t, uint64_t>(self);
  self_info.collapseDims();

  // generate bernoulli distribution
  auto rand_ptr = rand.data_ptr<int32_t>();
  int64_t numel = self.numel();
  std::initializer_list<std::uint64_t> seed = {seeds.first, 0, seeds.second};
  float val = static_cast<float>(accscalar_t(1) - p);

#ifdef USE_ONEMKL
  oneapi::mkl::rng::philox4x32x10 engine(sycl_queue, seed);
  oneapi::mkl::rng::bernoulli<scalar_t> distr(val);
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
  at::Tensor rand = at::empty_like(self, self.suggest_memory_format());
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

template <typename scalar_t, typename accscalar_t>
void fused_dropout_kernel(
    const Tensor& self,
    const Tensor& rand,
    Tensor& ret,
    Tensor& mask,
    accscalar_t p) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto self_info = getTensorInfo<scalar_t, uint64_t>(self);
  self_info.collapseDims();
  auto rand_ptr = rand.data_ptr<scalar_t>();
  int64_t numel = self.numel();

  accscalar_t pinv = accscalar_t(1) / p;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_ptr = self.data_ptr<scalar_t>();
    auto ret_ptr = ret.data_ptr<scalar_t>();
    auto mask_ptr = mask.data_ptr<uint8_t>();

    // There's big performance gap between w and w/o IndexToOffset,
    // thus we use two pathes to handle different cases
    if (self.is_contiguous()) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto index = item_id.get_linear_id();
        auto rv = accscalar_t(rand_ptr[index]) < p;
        if (index < numel) {
          ret_ptr[index] = self_ptr[index] * rv * pinv;
          mask_ptr[index] = (uint8_t)rv;
        }
      };
      cgh.parallel_for(DPCPP::range<1>(numel), kfn);
    } else {
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
    }
  };

  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

std::tuple<Tensor, Tensor> _fused_dropout_impl(
    const Tensor& self,
    const Tensor& rand,
    double p) {
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
        impl::fused_dropout_kernel<scalar_t, accscalar_t>(
            self, rand, ret, mask, pa);
      });

  return std::tuple<Tensor, Tensor>(ret, mask);
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

    // There's big performance gap between w and w/o IndexToOffset,
    // thus we use two pathes to handle different cases
    if (self.is_contiguous() && mask.is_contiguous()) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto index = item_id.get_linear_id();
        ret_ptr[index] =
            self_ptr[index] * static_cast<uint8_t>(mask_ptr[index]) * scale;
      };
      cgh.parallel_for(DPCPP::range<1>(numel), kfn);
    } else {
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
    }
  };

  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

} // namespace impl

std::tuple<Tensor, Tensor> _fused_dropout(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen_) {
#ifdef USE_ONEMKL
  Tensor rand = bernoulliDistr_impl(self, p, gen_);
#else
  at::Tensor rand = at::empty_like(self, self.suggest_memory_format());
  rand.bernoulli_(1 - p);
  rand.div_(1 - p);
#endif

  return impl::_fused_dropout_impl(self, rand, p);
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
        impl::masked_scale_kernel<scalar_t, accscalar_t>(ret, self, mask, pa);
      });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
