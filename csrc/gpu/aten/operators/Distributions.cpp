#include <ATen/native/Distributions.h>
#include <core/Generator.h>
#include <utils/DPCPP.h>
#include "DistributionTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

struct rand_uniform_wrapper {
  at::AtenIpexTypeXPU::randStatePhilox4_32_10_t& state;
  rand_uniform_wrapper(at::AtenIpexTypeXPU::randStatePhilox4_32_10_t& state)
      : state(state) {}
  float operator()() {
    uint32_t val = rand(&state); // need just bits
    constexpr auto MASK = static_cast<uint32_t>(
        (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    constexpr auto DIVISOR = static_cast<float>(1) /
        (static_cast<uint32_t>(1) << std::numeric_limits<float>::digits);
    return (val & MASK) * DIVISOR;
  }
};

template <typename scalar_t>
void binomial_kernel_dpcpp(
    at::Tensor& ret,
    const at::Tensor& count,
    const at::Tensor& prob,
    at::AtenIpexTypeXPU::PhiloxState philox_args) {
  using accscalar_t = at::AtenIpexTypeXPU::acc_type<scalar_t>;
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .add_output(ret)
                                .add_input(count)
                                .add_input(prob)
                                .build();

  at::AtenIpexTypeXPU::distribution_binary_kernel(
      iter,
      philox_args,
      [&](at::AtenIpexTypeXPU::randStatePhilox4_32_10_t& state,
          scalar_t count,
          scalar_t prob) {
        auto uniform_lambda = rand_uniform_wrapper(state);
        BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(
            uniform_lambda);
        auto sample =
            sample_binomial<scalar_t, accscalar_t, decltype(uniform_lambda)>(
                count, prob, standard_uniform);
        return static_cast<scalar_t>(sample);
      });
}

template <typename scalar_t>
void gamma_kernel_dpcpp(
    at::Tensor& ret,
    const at::Tensor& alpha,
    at::AtenIpexTypeXPU::PhiloxState philox_args) {
  using accscalar_t = at::AtenIpexTypeXPU::acc_type<scalar_t>;
  at::TensorIterator iter =
      at::TensorIteratorConfig().add_output(ret).add_input(alpha).build();
  auto functor = [philox_args](
                     at::AtenIpexTypeXPU::randStatePhilox4_32_10_t& state,
                     scalar_t& ret_val,
                     const scalar_t& alpha) {
    auto seeds = philox_unpack(philox_args);

    auto uniform_lambda = [&state]() { return rand_uniform(&state); };
    BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(
        uniform_lambda);

    auto normal_lambda = [&state]() { return rand_normal(&state); };
    BaseSampler<accscalar_t, decltype(normal_lambda)> standard_normal(
        normal_lambda);
    auto sample = sample_gamma<
        scalar_t,
        accscalar_t,
        decltype(uniform_lambda),
        decltype(normal_lambda)>(alpha, standard_uniform, standard_normal);
    auto min_value = std::numeric_limits<scalar_t>::min();
    ret_val = (min_value > sample) ? min_value : sample;
  };
  at::AtenIpexTypeXPU::
      distribution_unary_kernel<scalar_t, scalar_t, decltype(functor)>(
          iter, philox_args, functor);
}

} // namespace impl

Tensor binomial(
    const Tensor& count,
    const Tensor& prob,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    engine_inputs = gen->philox_engine_inputs(42);
  }
  PhiloxState rng_engine_inputs(
      std::get<0>(engine_inputs), std::get<1>(engine_inputs));
  Tensor ret = at::empty(count.sizes(), count.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      ret.scalar_type(), "binomial_xpu", [&]() {
        impl::binomial_kernel_dpcpp<scalar_t>(
            ret, count, prob, rng_engine_inputs);
      });
  return ret;
}

Tensor _standard_gamma(const Tensor& alpha, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    engine_inputs = gen->philox_engine_inputs(10);
  }
  PhiloxState rng_engine_inputs(
      std::get<0>(engine_inputs), std::get<1>(engine_inputs));
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "gamma_xpu",
      [&]() {
        impl::gamma_kernel_dpcpp<scalar_t>(ret, alpha, rng_engine_inputs);
      });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
