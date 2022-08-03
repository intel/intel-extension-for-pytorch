#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Distributions.h"
#include "Loops.h"
#include "Random.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& exponential_(
    Tensor& self,
    double lambda,
    c10::optional<Generator> generator) {
  TORCH_CHECK(
      lambda >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambda);
  auto gen = get_generator_or_default<DPCPPGeneratorImpl>(
      generator, getDefaultDPCPPGenerator());
#ifdef USE_ONEMKL
  if (lambda > 0 && lambda < std::numeric_limits<double>::max() &&
      self.is_contiguous() && self.scalar_type() == at::ScalarType::Float) {
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "exponential_dpcpp_", [&] {
      scalar_t displ = static_cast<scalar_t>(0.0);
      scalar_t scale = static_cast<scalar_t>(std::abs(1 / lambda));
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      uint64_t seed;
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        seed = gen->current_seed();
      }
      oneapi::mkl::rng::philox4x32x10 engine(dpcpp_queue, seed);
      oneapi::mkl::rng::exponential<scalar_t> distribution(displ, scale);
      DPCPP_ONEMKL_SUBMIT(
          dpcpp_queue,
          oneapi::mkl::rng::generate,
          distribution,
          engine,
          self.numel(),
          (scalar_t*)(self.data_ptr()));
    });
  } else
#endif
  {
    auto iter = TensorIterator::nullary_op(self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "exponential_dpcpp_",
        [&]() {
          // IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(),
          // "exponential_dpcpp_", [&] {
          using accscalar_t = acc_type<scalar_t>;
          auto lambd = static_cast<accscalar_t>(lambda);

          // define lambd for exponential transformation
          auto exponential_func = [lambd](accscalar_t rand) {
            accscalar_t sample;
            sample = Numerics<accscalar_t>::log(rand);
            return static_cast<scalar_t>(
                static_cast<accscalar_t>(-1.0) / lambd * sample);
          };
          distribution_nullary_kernel<scalar_t, accscalar_t>(
              iter,
              gen,
              [](RandomState<Philox4_32_10>* state) {
                return state->uniform<scalar_t>();
              },
              exponential_func);
        });
  }

  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
