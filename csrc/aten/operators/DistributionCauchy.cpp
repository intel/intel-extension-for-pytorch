#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include <utils/oneMKLUtils.h>
#include "Distributions.h"
#include "Random.h"

namespace at {
namespace AtenIpexTypeXPU {

void cauchy_kernel(
    TensorIterator& iter,
    double median_,
    double sigma_,
    c10::optional<Generator> generator) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      generator, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "cauchy_dpcpp",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        auto median = static_cast<accscalar_t>(median_);
        auto sigma = static_cast<accscalar_t>(sigma_);
        auto cauchy_func = [median, sigma](accscalar_t rand) {
          return static_cast<scalar_t>(
              median +
              sigma *
                  at::tan(
                      Numerics<accscalar_t>::pi() *
                      (rand - static_cast<accscalar_t>(0.5))));
        };
        AtenIpexTypeXPU::distribution_nullary_kernel<scalar_t, accscalar_t>(
            iter,
            gen,
            [](RandomState<Philox4_32_10>* state) {
              return state->uniform<scalar_t>();
            },
            cauchy_func);
      });
}

#ifdef USE_ONEMKL
template <typename scalar_t, typename accscalar_t>
void cauchy_mkl_kernel(
    Tensor& self,
    int64_t numel,
    accscalar_t median_,
    accscalar_t sigma_,
    c10::optional<Generator> generator) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      generator, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }
  std::initializer_list<std::uint64_t> seed = {
      rng_engine_inputs.first, 0, rng_engine_inputs.second};
  auto& sycl_queue = dpcppGetCurrentQueue();

  auto self_ptr = self.data_ptr<scalar_t>();
  auto median = static_cast<double>(median_);
  auto sigma = static_cast<double>(sigma_);
  oneapi::mkl::rng::philox4x32x10 engine(sycl_queue, seed);
  oneapi::mkl::rng::cauchy<scalar_t> distr(median, sigma);
  oneapi::mkl::rng::generate(distr, engine, numel, self_ptr);
}
#endif

Tensor& cauchy_(
    Tensor& self,
    double median,
    double sigma,
    c10::optional<Generator> generator) {
#ifdef USE_ONEMKL
  int64_t numel = self.numel();
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "cauchy_mkl_dpcpp", [&] {
    using accscalar_t = DiscreteDistributionType<scalar_t>::type;
    auto median_ = static_cast<accscalar_t>(median);
    auto sigma_ = static_cast<accscalar_t>(sigma);
    cauchy_mkl_kernel<scalar_t>(self, numel, median_, sigma_, generator);
  });
#else
  auto iter = TensorIterator::nullary_op(self);
  cauchy_kernel(iter, median, sigma, generator);
#endif
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
