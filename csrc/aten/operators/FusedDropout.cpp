#include <ATen/ATen.h>

#include <core/Generator.h>
#include <runtime/Utils.h>
#include "comm/ApplyUtils.h"
#include "comm/ATDispatch.h"
#include "Distributions.h"

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

// TODO: use operator DistributionBernoulli to replace this function when performance is ready.
template<typename scalar_t>
void bernoulliDistr(scalar_t* rand, 
    std::pair<uint64_t, uint64_t> seeds,
    int64_t numel,
    float p) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  std::initializer_list<std::uint64_t> seed = { seeds.first, 0, seeds.second };
#ifdef USE_ONEMKL
  oneapi::mkl::rng::philox4x32x10 engine(sycl_queue, seed);
  oneapi::mkl::rng::bernoulli<scalar_t> distr(p); 
  oneapi::mkl::rng::generate(distr, engine, numel, rand);
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

template <typename...>
class contigu_fused_dropout_kernel_dpcpp {};
template <typename...>
class fused_dropout_kernel_dpcpp {};
template<typename scalar_t, typename accscalar_t>
void fused_dropout_kernel(const Tensor& self,
    Tensor& ret,
    Tensor& mask,
    accscalar_t p,
    std::pair<uint64_t, uint64_t> seeds) {
  
  auto &sycl_queue = dpcppGetCurrentQueue();
  auto self_info = getTensorInfo<scalar_t, uint64_t>(self);
  self_info.collapseDims();
  
  // generate bernoulli distribution
  float val = static_cast<float>(accscalar_t(1) - p);
  at::Tensor rand=at::empty(self.sizes(), self.options().dtype(kInt), self.suggest_memory_format());
  auto rand_ptr = rand.data_ptr<int32_t>();
  int64_t numel = self.numel();
  bernoulliDistr<int32_t>(rand_ptr, seeds, numel, val); 

  // implement dropout result and mask according to distr
  accscalar_t pinv = accscalar_t(1) / p;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto self_ptr = self.data_ptr<scalar_t>();
    auto ret_ptr = ret.data_ptr<scalar_t>();
    auto mask_ptr = mask.data_ptr<uint8_t>();

    // There's big performance gap between w and w/o IndexToOffset, 
    // thus we use two pathes to handle different cases
    if (self.is_contiguous()) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id)  {
        auto index = item_id.get_linear_id();
        auto rv = rand_ptr[index] < p;
        if (index < numel) {
          ret_ptr[index] = self_ptr[index] * rv * pinv;
          mask_ptr[index] = (uint8_t)rv;
        }
      };
      cgh.parallel_for<contigu_fused_dropout_kernel_dpcpp<scalar_t, accscalar_t>>(
        DPCPP::range<1>(numel),
        kfn);
    } else {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id)  {
        auto index = item_id.get_linear_id();
        auto self_offset = IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
        auto rv = rand_ptr[index] < p;
        if (index < numel) {
          ret_ptr[index] = self_ptr[self_offset] * rv * pinv;
          mask_ptr[index] = (uint8_t)rv;
        }
      };
      cgh.parallel_for<fused_dropout_kernel_dpcpp<scalar_t, accscalar_t>>(
        DPCPP::range<1>(numel),
        kfn);
    }
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

template <typename...>
class contigu_masked_scale_ker {};
template <typename...>
class masked_scale_ker {};
template<typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const Tensor & self, const Tensor mask, accscalar_t scale){
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
        ret_ptr[index] = self_ptr[index] * mask_ptr[index] * scale;
      };
      cgh.parallel_for<contigu_masked_scale_ker<scalar_t, accscalar_t>>(
        DPCPP::range<1>(numel),
        kfn);
    } else {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto index = item_id.get_linear_id();
        auto self_offset = IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
        auto mask_offset = IndexToOffset<uint8_t, uint64_t>::get(index, mask_info);
        ret_ptr[index] = self_ptr[self_offset] * mask_ptr[mask_offset] * scale;
      };
      cgh.parallel_for<masked_scale_ker<scalar_t, accscalar_t>>(
        DPCPP::range<1>(numel),
        kfn);
    }
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

} // namespace::impl

std::tuple<Tensor,Tensor> _fused_dropout(const Tensor& self, double p, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(gen_, getDefaultDPCPPGenerator());

  Tensor ret = at::empty_like(self, self.suggest_memory_format());
  Tensor mask = at::empty(self.sizes(), self.options().dtype(kByte), self.suggest_memory_format());
  int64_t nelem = self.numel();
  //empty tensors should not get here, but just in case, avoid FPE
  if (nelem==0) return std::tuple<Tensor,Tensor>(self, mask); 
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }
  
  // TODO: Should add path for not satisfy canUse32BitIndexMath
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "dropout_kernel", [&] {
    using accscalar_t = DiscreteDistributionType<scalar_t>::type; 
    accscalar_t pa = (accscalar_t)(p);
    impl::fused_dropout_kernel<scalar_t>(self, ret, mask, pa, rng_engine_inputs);
  });
  return std::tuple<Tensor,Tensor>(ret, mask);
}

Tensor _masked_scale(const Tensor& self, const Tensor& mask, double scale) {
  Tensor ret = at::empty_like(self, self.suggest_memory_format());
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "masked_scale", [&] {
    using accscalar_t = DiscreteDistributionType<scalar_t>::type; 
    accscalar_t pa = (accscalar_t)(scale);
    impl::masked_scale_kernel<scalar_t>(ret, self, mask, pa);
  });
  return ret;
}

}} // namespace at::AtenIpexTypeXPU
