//
// Created by johnlu on 2020/9/2.
//

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <core/Generator.h>
#include <utils/ATDispatch.h>

#include <ATen/aten_ipex_type_dpcpp.h>
#include "Random.h"
#include "Distributions.h"


namespace at {
namespace AtenIpexTypeDPCPP {

DPCPP_DEF_K1(bernoulli_tensor_dpcpp_ker);

Tensor& bernoulli_(Tensor& self, const Tensor& p_, Generator* gen_) {
  at::AtenIpexTypeDPCPP::uniform_(self, 0.0, 1.0, gen_);
  auto p = p_.to(kDPCPP);
  auto iter = TensorIterator::binary_op(self, self, p);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "bernoulli_tensor_dpcpp_", [&] {
    dpcpp_kernel_for_tensor_iter<DPCPP_K(bernoulli_tensor_dpcpp_ker)>(
      iter, [](scalar_t self, scalar_t p) -> scalar_t { return static_cast<scalar_t>(self < p); });
  });
  return self;
}

void bernoulli_scalar_dpcpp(TensorIterator& iter, double p_, Generator* gen_) {
  auto gen = get_generator_or_default<DPCPPGenerator>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "bernoulli_scalar_dpcpp", [&] {
    using accscalar_t = DiscreteDistributionType<scalar_t>::type;
    auto p = static_cast<accscalar_t>(p_);
    // define lambda for bernoulli transformation
    auto bernoulli_func = [p] (accscalar_t rand) {
      return static_cast<scalar_t>(rand < static_cast<accscalar_t>(p));
    };
    AtenIpexTypeDPCPP::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
      gen,
      [] (RandomState<Philox4_32_10>* state) { return state->uniform<scalar_t>(); },
      bernoulli_func);
  });
}

Tensor& bernoulli_(Tensor& self, double p, Generator* gen_) {
  auto iter = TensorIterator::nullary_op(self);
  bernoulli_scalar_dpcpp(iter, p, gen_);
  return self;
}

}} // namespace at::AtenIpexTypeDPCPP