#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include <utils/ATDispatch.h>
#include <utils/AccumulateType.h>

#include "Random.h"
#include "Distributions.h"

namespace at {
namespace AtenIpexTypeDPCPP {

void normal_dpcpp(TensorIterator& iter, double mean_, double std_, Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_dpcpp", [&] {
    using accscalar_t = dist_acctype<scalar_t>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto normal_func = [mean, std] (accscalar_t rand) {
      auto ret = static_cast<scalar_t>(rand * std + mean);
      return ret;
    };
    AtenIpexTypeDPCPP::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
      gen,
      [] (RandomState<Philox4_32_10>* state) { return state->normal<scalar_t>(); },
      normal_func);
  });


}

Tensor& normal_(Tensor& self, double mean, double std, Generator* generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  normal_dpcpp(iter, mean, std, generator);
  return self;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at