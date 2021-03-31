#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/DistributionsHelper.h>

#include <core/Generator.h>
#include <utils/ATDispatch.h>
#include <utils/AccumulateType.h>

#include "Random.h"
#include "Distributions.h"

namespace at {
namespace AtenIpexTypeXPU {


template<typename RNG>
void random_kernel(TensorIterator& iter, c10::optional<RNG> gen_) {
  auto gen = get_generator_or_default<at::DPCPPGeneratorImpl>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_kernel_fp", [&] {
      if (std::is_same<scalar_t, double>::value) {
        auto random_func = [] (uint64_t rand) {
          return transformation::uniform_int<scalar_t>(rand);
        };
        distribution_nullary_kernel<scalar_t, uint64_t>(iter,
          gen,
          [] (RandomState<Philox4_32_10>* state) { return state-> random<uint64_t>(); },
          random_func);
      } else {
        auto random_func = [] (uint32_t rand) {
          return transformation::uniform_int<scalar_t>(rand);
        };
        distribution_nullary_kernel<scalar_t, uint32_t>(iter,
          gen,
          [] (RandomState<Philox4_32_10>* state) { return state-> random<uint32_t>(); },
          random_func);
      }
    });
  } else if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = typename c10::impl::ScalarTypeToCPPType<ScalarType::Bool>::type;
    auto random_func = [] (uint32_t rand) {
      return transformation::uniform_int<scalar_t>(rand);
    };
    distribution_nullary_kernel<scalar_t, uint32_t>(iter,
      gen,
      [] (RandomState<Philox4_32_10>* state) { return state-> random<uint32_t>(); },
      random_func);
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "random_kernel_int", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        auto random_func = [] (uint64_t rand) {
          return transformation::uniform_int<scalar_t>(rand);
        };
        distribution_nullary_kernel<scalar_t, uint64_t>(iter,
          gen,
          [] (RandomState<Philox4_32_10>* state) { return state-> random<uint64_t>(); },
          random_func);
      } else {
        auto random_func = [] (uint32_t rand) {
          return transformation::uniform_int<scalar_t>(rand);
        };
        distribution_nullary_kernel<scalar_t, uint32_t>(iter,
          gen,
          [] (RandomState<Philox4_32_10>* state) { return state-> random<uint32_t>(); },
          random_func);
      }
    });    
  } else {
    TORCH_CHECK(false, "random_kernel handles only integral, floating-point and boolean types");
  }
}

template<typename RNG>
void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<RNG> gen_) {
  auto gen = get_generator_or_default<at::DPCPPGeneratorImpl>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());    
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel", [&] {
    if ((
      std::is_same<scalar_t, int64_t>::value ||
      std::is_same<scalar_t, double>::value ||
      std::is_same<scalar_t, float>::value ||
      std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
    {
      // define lambda to mod with range and add base
      auto random_func = [range, base] (uint64_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      distribution_nullary_kernel<scalar_t, uint64_t>(iter,
        gen,
        [] (RandomState<Philox4_32_10>* state) { return state-> random<uint64_t>(); },
        random_func);
    } else {
      auto random_func = [range, base] (uint32_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      distribution_nullary_kernel<scalar_t, uint32_t>(iter,
        gen,
        [] (RandomState<Philox4_32_10>* state) { return state-> random<uint32_t>(); },
        random_func);
    }
  });
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIterator& iter, c10::optional<RNG> gen_) {
  auto gen = get_generator_or_default<at::DPCPPGeneratorImpl>(gen_, dpcpp::detail::getDefaultDPCPPGenerator());    
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel", [&] {
    if (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      auto random_func = [] (uint64_t rand) {
        return transformation::uniform_int_full_range<scalar_t>(rand);
      };
      distribution_nullary_kernel<scalar_t, uint64_t>(iter,
        gen,
        [] (RandomState<Philox4_32_10>* state) { return state-> random<uint64_t>(); },
        random_func);
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomDPCPPStub {
  void operator()(TensorIterator& iter, c10::optional<RNG> gen) {
    random_kernel(iter, gen);
  }
};

Tensor & random_(Tensor & self, c10::optional<Generator> gen_){
  return at::native::templates::random_impl<RandomDPCPPStub, Generator>(self, gen_);
}

template<typename RNG>
struct RandomFromToDPCPPStub {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<RNG> gen) {
    random_from_to_kernel(iter, range, base, gen);
  }
  void operator()(TensorIterator& iter, c10::optional<RNG> gen) {
    random_full_64_bits_range_kernel(iter, gen);
  }
};

Tensor & random_(Tensor & self, int64_t from, optional<int64_t> to, c10::optional<Generator> gen_){
  return at::native::templates::random_from_to_impl<RandomFromToDPCPPStub, Generator>(self, from, to, gen_);
}

Tensor & random_(Tensor & self, int64_t to, c10::optional<Generator> gen_){
  return random_(self, 0, to, gen_);
}

} // namespace AtenIpexTypeXPU
} // namespace at

