#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename uint_t>
struct random_kernel_random_func_functor {
  auto operator()(uint_t rand) const {
    return transformation::uniform_int<scalar_t>(rand);
  }
};

struct random_kernel_random_state_functor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    rand_vec2<uint64_t> ret;
    auto rand_val = rand4(state);
    ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
    ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
    return ret;
  }
};

struct random_kernel_random_state_functor_2 {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand4(state);
  }
};

template <typename RNG>
void random_kernel(TensorIterator& iter, c10::optional<RNG> gen_) {
  auto gen =
      get_generator_or_default<torch_ipex::xpu::dpcpp::DPCPPGeneratorImpl>(
          gen_, torch_ipex::xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  if (isFloatingType(iter.dtype())) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "random_kernel_fp",
        [&] {
          if (std::is_same<scalar_t, double>::value) {
            random_kernel_random_state_functor random_state_func;
            random_kernel_random_func_functor<scalar_t, uint64_t> random_func;
            distribution_nullary_kernel<
                scalar_t,
                uint64_t,
                PHILOX_ENGINE_CALLS / 2>(
                iter, gen, random_state_func, random_func);
          } else {
            random_kernel_random_state_functor_2 random_state_func;
            random_kernel_random_func_functor<scalar_t, uint32_t> random_func;
            distribution_nullary_kernel<
                scalar_t,
                uint32_t,
                PHILOX_ENGINE_CALLS>(iter, gen, random_state_func, random_func);
          }
        });
  } else if (iter.dtype() == ScalarType::Bool) {
    using scalar_t =
        typename c10::impl::ScalarTypeToCPPType<ScalarType::Bool>::type;
    random_kernel_random_func_functor<scalar_t, uint32_t> random_func;
    random_kernel_random_state_functor_2 random_state_func;
    distribution_nullary_kernel<scalar_t, uint32_t, PHILOX_ENGINE_CALLS>(
        iter, gen, random_state_func, random_func);
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "random_kernel_int", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        random_kernel_random_state_functor random_state_func;
        random_kernel_random_func_functor<scalar_t, uint64_t> random_func;
        distribution_nullary_kernel<
            scalar_t,
            uint64_t,
            PHILOX_ENGINE_CALLS / 2>(iter, gen, random_state_func, random_func);
      } else {
        random_kernel_random_state_functor_2 random_state_func;
        random_kernel_random_func_functor<scalar_t, uint32_t> random_func;
        distribution_nullary_kernel<scalar_t, uint32_t, PHILOX_ENGINE_CALLS>(
            iter, gen, random_state_func, random_func);
      }
    });
  } else {
    TORCH_CHECK(
        false,
        "random_kernel handles only integral, floating-point and boolean types");
  }
}

template <typename scalar_t, typename uint_t>
struct random_from_to_kernel_functor {
  auto operator()(uint_t rand) const {
    return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
  }
  random_from_to_kernel_functor(uint64_t range, int64_t base)
      : range(range), base(base) {}

 private:
  uint64_t range;
  int64_t base;
};

template <typename RNG>
void random_from_to_kernel(
    TensorIterator& iter,
    uint64_t range,
    int64_t base,
    c10::optional<RNG> gen_) {
  auto gen =
      get_generator_or_default<torch_ipex::xpu::dpcpp::DPCPPGeneratorImpl>(
          gen_, torch_ipex::xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "random_from_to_kernel",
      [&] {
        if ((std::is_same<scalar_t, int64_t>::value ||
             std::is_same<scalar_t, double>::value ||
             std::is_same<scalar_t, float>::value ||
             std::is_same<scalar_t, at::BFloat16>::value) &&
            range >= 1ULL << 32) {
          // define lambda to mod with range and add base
          random_kernel_random_state_functor random_state_func;
          random_from_to_kernel_functor<scalar_t, uint64_t> random_func(
              range, base);
          distribution_nullary_kernel<
              scalar_t,
              uint64_t,
              PHILOX_ENGINE_CALLS / 2>(
              iter, gen, random_state_func, random_func);
        } else {
          random_kernel_random_state_functor_2 random_state_func;
          random_from_to_kernel_functor<scalar_t, uint32_t> random_func(
              range, base);
          distribution_nullary_kernel<scalar_t, uint32_t, PHILOX_ENGINE_CALLS>(
              iter, gen, random_state_func, random_func);
        }
      });
}

template <typename scalar_t>
struct random_full_64_bits_range_kernel_functor {
  auto operator()(uint64_t rand) const {
    return transformation::uniform_int_full_range<scalar_t>(rand);
  }
};

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template <typename RNG>
void random_full_64_bits_range_kernel(
    TensorIterator& iter,
    c10::optional<RNG> gen_) {
  auto gen =
      get_generator_or_default<torch_ipex::xpu::dpcpp::DPCPPGeneratorImpl>(
          gen_, torch_ipex::xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "random_full_64_bits_range_kernel",
      [&] {
        if (std::is_same<scalar_t, int64_t>::value ||
            std::is_same<scalar_t, double>::value ||
            std::is_same<scalar_t, float>::value ||
            std::is_same<scalar_t, at::BFloat16>::value) {
          random_kernel_random_state_functor random_state_func;
          random_full_64_bits_range_kernel_functor<scalar_t> random_func;
          distribution_nullary_kernel<
              scalar_t,
              uint64_t,
              PHILOX_ENGINE_CALLS / 2>(
              iter, gen, random_state_func, random_func);
        } else {
          TORCH_CHECK(
              false,
              "random_full_64_bits_range_kernel handles only int64, double, float and bfloat16");
        }
      });
}

template <typename RNG>
struct RandomDPCPPStub {
  void operator()(TensorIterator& iter, c10::optional<RNG> gen) {
    random_kernel(iter, gen);
  }
};

Tensor& random_(Tensor& self, c10::optional<Generator> gen_) {
  return at::native::templates::random_impl<RandomDPCPPStub, Generator>(
      self, gen_);
}

template <typename RNG>
struct RandomFromToDPCPPStub {
  void operator()(
      TensorIterator& iter,
      uint64_t range,
      int64_t base,
      c10::optional<RNG> gen) {
    random_from_to_kernel(iter, range, base, gen);
  }
  void operator()(TensorIterator& iter, c10::optional<RNG> gen) {
    random_full_64_bits_range_kernel(iter, gen);
  }
};

Tensor& random_(
    Tensor& self,
    int64_t from,
    optional<int64_t> to,
    c10::optional<Generator> gen_) {
  return at::native::templates::
      random_from_to_impl<RandomFromToDPCPPStub, Generator>(
          self, from, to, gen_);
}

Tensor& random_(Tensor& self, int64_t to, c10::optional<Generator> gen_) {
  return random_(self, 0, to, gen_);
}

} // namespace AtenIpexTypeXPU
} // namespace at
