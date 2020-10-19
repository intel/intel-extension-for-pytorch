#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <utils/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#include "Random.h"
#include "Loops.h"
#include "Distributions.h"

#ifdef USE_ONEMKL
#include <mkl_sycl.hpp>
#include <mkl.h>
#endif

using namespace at::dpcpp::detail;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor& exponential_(Tensor& self, double lambda_, Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
#ifdef USE_ONEMKL
  if (lambda_ > 0 && self.is_contiguous()) {
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "exponential_dpcpp_", [&] {
      scalar_t displ = static_cast<scalar_t>(0.0);
      scalar_t scale = static_cast<scalar_t>(std::abs(1/lambda_));
      auto &sycl_queue = dpcpp::getCurrentDPCPPStream().dpcpp_queue();
      uint64_t seed;
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        seed = gen->current_seed();
      }
      mkl::rng::philox4x32x10 engine(sycl_queue, seed);
      mkl::rng::exponential<scalar_t> distribution(displ, scale);
      auto sycl_buffer = make_buffer<scalar_t>(self.data_ptr());
      mkl::rng::generate(distribution, engine, self.numel(), sycl_buffer);
    });
  } else
#endif
  {
    auto iter = TensorIterator::nullary_op(self);
    IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "exponential_dpcpp_", [&] {
      using accscalar_t = acc_type<scalar_t>;
      auto lambda = static_cast<accscalar_t>(lambda_);

      // define lambda for exponential transformation
      auto exponential_func = [lambda] (accscalar_t rand) {
        accscalar_t sample;
        sample = DPCPP::log(rand);
        return static_cast<scalar_t>(static_cast<accscalar_t>(-1.0) / lambda * sample);
      };
      distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
                                                         gen,
                                                         [] (RandomState<Philox4_32_10>* state) {
                                                           return state->uniform<scalar_t>(); },
                                                         exponential_func);

    });
  }

  return self;
}


} // namespace AtenIpexTypeDPCPP
} // namespace at
