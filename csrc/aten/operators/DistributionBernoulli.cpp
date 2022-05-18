#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "Distributions.h"
#include "Random.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor& uniform_(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> generator);

Tensor& bernoulli_(
    Tensor& self,
    const Tensor& p_,
    c10::optional<Generator> gen_) {
  Tensor self_float;
  auto self_type = self.scalar_type();
  if (!(self_type == at::ScalarType::Float ||
        self_type == at::ScalarType::Double))
    self_float = at::empty(self.sizes(), self.options().dtype(at::kFloat));
  else
    self_float = self;
  at::AtenIpexTypeXPU::uniform_(self_float, 0.0, 1.0, gen_);
  auto p = p_.to(kXPU);
  auto iter = TensorIteratorConfig()
                  .add_output(self)
                  .add_input(self_float)
                  .add_input(p)
                  .check_all_same_dtype(false)
                  .build();

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "bernoulli_tensor_dpcpp_",
      [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t self_float, scalar_t p) -> scalar_t {
              return static_cast<scalar_t>(self_float < p);
            });
      });
  return self;
}

void bernoulli_scalar_dpcpp(
    TensorIterator& iter,
    double p_,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "bernoulli_scalar_dpcpp",
      [&] {
        using accscalar_t = DiscreteDistributionType<scalar_t>::type;
        auto p = static_cast<accscalar_t>(p_);
        // define lambda for bernoulli transformation
        auto bernoulli_func = [p](accscalar_t rand) {
          return static_cast<scalar_t>(rand < static_cast<accscalar_t>(p));
        };
        if (std::is_same<scalar_t, double>::value) {
          AtenIpexTypeXPU::distribution_nullary_kernel<scalar_t, accscalar_t>(
              iter,
              gen,
              [](RandomState<Philox4_32_10>* state) {
                return state->uniform<double>();
              },
              bernoulli_func);
        } else {
          AtenIpexTypeXPU::distribution_nullary_kernel<scalar_t, accscalar_t>(
              iter,
              gen,
              [](RandomState<Philox4_32_10>* state) {
                return state->uniform<float>();
              },
              bernoulli_func);
        }
      });
}

Tensor& bernoulli_(Tensor& self, double p, c10::optional<Generator> gen_) {
  auto iter = TensorIterator::nullary_op(self);
  bernoulli_scalar_dpcpp(iter, p, gen_);
  return self;
}

Tensor& bernoulli_out(
    const Tensor& self,
    c10::optional<Generator> generator,
    Tensor& out) {
  auto out_type = out.scalar_type();
  Tensor out_float;
  if (!(out_type == at::ScalarType::Float ||
        out_type == at::ScalarType::Double))
    out_float = self.to(at::ScalarType::Float);
  else
    out_float = out;
  at::AtenIpexTypeXPU::uniform_(out_float, 0.0, 1.0, generator);
  auto iter = TensorIteratorConfig()
                  .add_output(out)
                  .add_input(out_float)
                  .add_input(self)
                  .check_all_same_dtype(false)
                  .build();
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "bernoulli_tensor_dpcpp_",
      [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t out, scalar_t p) -> scalar_t {
              return static_cast<scalar_t>(out < p);
            });
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
