#pragma once

#include <ATen/core/PhiloxRNGEngine.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "MemoryAccess.h"
#include "RandomEngine.h"
#include "comm/Numerics.h"
#include "core/detail/OffsetCalculator.h"

namespace at {
namespace AtenIpexTypeXPU {

#define PHILOX_ENGINE_CALLS 4

struct PhiloxState {
  PhiloxState() = default;
  // Called if graph capture is not underway
  PhiloxState(uint64_t seed, uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxState(
      uint64_t seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_ = 0;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

inline std::tuple<uint64_t, uint64_t> philox_unpack(PhiloxState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to
    // "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire
    // kernel. For most threads' reads it will hit in cache, so it shouldn't
    // hurt performance.
    return std::make_tuple(
        arg.seed_,
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

// Just follow loops.h design
template <
    typename accscalar_t,
    int unroll_factor,
    typename dist_t,
    typename transform_t,
    typename item_t>
inline void distribution_elementwise_kernel(
    item_t& item,
    int numel,
    PhiloxState philox_args,
    const dist_t dist_func,
    const transform_t transform_func) {
  int group_size = item.get_local_range(0);
  int num_groups = item.get_group_range(0);
  int idx = item.get_group(0) * group_size + item.get_local_id(0);

  auto seeds = philox_unpack(philox_args);
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int full_tile_work_size = group_size * num_groups * unroll_factor;
  int rounded_size =
      ((numel - 1) / full_tile_work_size + 1) * full_tile_work_size;
  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += full_tile_work_size) { // global range stride
    auto rand = dist_func(&state);
#pragma unroll
    for (int i = 0; i < unroll_factor; i++) {
      int li = linear_index + group_size * num_groups * i;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[i]));
      }
    }
    // Some state (e.g. MTGP32) need to add barrier there.
  }
}

inline std::tuple<uint64_t, uint32_t, uint32_t> calc_execution_policy(
    int64_t total_elements) {
  auto group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  auto num_groups = (total_elements + group_size - 1) / group_size;
  auto hw_max_groups = dpcppMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  uint64_t counter_offset =
      ((total_elements - 1) / (group_size * num_groups * PHILOX_ENGINE_CALLS) +
       1) *
      PHILOX_ENGINE_CALLS;
  return std::make_tuple(counter_offset, num_groups, group_size);
}

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename RNG,
    typename dist_t,
    typename transform_t>
void distribution_nullary_kernel(
    at::TensorIteratorBase& iter,
    RNG gen,
    const dist_t& dist_func,
    const transform_t transform_func) {
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto execution_policy = calc_execution_policy(numel);
  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, unroll_factor>(
          sub_iter, gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  auto& sycl_queue = dpcppGetCurrentQueue();

  /* [Note: Why don't vectorize #1812]
   * Commit 36f9fab deletes the vectorization because the vectorization
   * acceleration would generate a different result with the default one. It is
   * nontrivial to get the same result of vectorized <-> non-vectorized. Thus,
   * if the user use a CUDA seed 1234, the vectorization result xpu_1234 output
   * would NOT be the identical value with cuda_1234. This makes user willing to
   * reproduce the best result with the given CUDA seed, would be difficult on
   * XPU platform.
   */
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_elementwise_kernel<accscalar_t, unroll_factor>(
            item,
            numel,
            PhiloxState(
                std::get<0>(rng_engine_inputs), std::get<1>(rng_engine_inputs)),
            dist_func,
            [=](int idx, accscalar_t rand) {
              scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
              *out = transform_func(rand);
            });
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_elementwise_kernel<accscalar_t, unroll_factor>(
            item,
            numel,
            PhiloxState(
                std::get<0>(rng_engine_inputs), std::get<1>(rng_engine_inputs)),
            dist_func,
            [=](int idx, accscalar_t rand) {
              auto offsets = offset_calc.get(idx);
              scalar_t* out = (scalar_t*)&out_data[offsets[0]];
              *out = transform_func(rand);
            });
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

// Unary kernel
template <
    typename scalar1_t,
    typename scalar2_t,
    typename func_t,
    typename inp_offset_calc_t,
    typename out_offset_calc_t,
    typename item_t>
void distribution_unary_elementwise_kernel(
    item_t& item,
    int numel,
    func_t f,
    PhiloxState philox_args,
    scalar1_t* output_data,
    const scalar2_t* input_data,
    inp_offset_calc_t inp_calc,
    out_offset_calc_t out_calc) {
  int group_size = item.get_local_range(0);
  int global_size = item.get_global_range(0);
  int idx = item.get_group(0) * group_size + item.get_local_id(0);

  auto seeds = philox_unpack(philox_args);
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int global_idx;
  for (int i = 0; i < numel; i += global_size) {
    global_idx = i + idx;
    auto in_offsets = inp_calc.get(global_idx);
    auto out_offsets = out_calc.get(global_idx);
    f(state, output_data[out_offsets[0]], input_data[in_offsets[0]]);
  }
}

template <typename scalar1_t, typename scalar2_t, typename func_t>
void distribution_unary_kernel(
    TensorIterator& iter,
    PhiloxState philox_args,
    const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_unary_kernel<scalar1_t, scalar2_t, decltype(f)>(
          sub_iter, philox_args, f);
    }
    return;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  int group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  int num_groups = (numel + group_size - 1) / group_size;
  int hw_max_groups = dpcppMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  scalar1_t* output_data = static_cast<scalar1_t*>(iter.data_ptr(0));
  const scalar2_t* input_data = static_cast<const scalar2_t*>(iter.data_ptr(1));

  auto& sycl_queue = dpcppGetCurrentQueue();

  if (iter.is_contiguous()) {
    auto input_offset_calculator = TrivialOffsetCalculator<1>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_unary_elementwise_kernel(
            item,
            numel,
            f,
            philox_args,
            output_data,
            input_data,
            input_offset_calculator,
            output_offset_calculator);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto input_offset_calculator = make_input_offset_calculator<1>(iter);
    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_unary_elementwise_kernel(
            item,
            numel,
            f,
            philox_args,
            output_data,
            input_data,
            input_offset_calculator,
            output_offset_calculator);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

// Binary kernel
template <
    typename func_t,
    typename inp_offset_calc_t,
    typename out_offset_calc_t,
    typename item_t>
void distribution_binary_elementwise_kernel(
    item_t& item,
    int numel,
    func_t f,
    PhiloxState philox_args,
    typename function_traits<func_t>::result_type* output_data,
    const typename function_traits<func_t>::template arg<1>::type* input_data_1,
    const typename function_traits<func_t>::template arg<2>::type* input_data_2,
    inp_offset_calc_t inp_calc,
    out_offset_calc_t out_calc) {
  int group_size = item.get_local_range(0);
  int global_size = item.get_global_range(0);
  int idx = item.get_group(0) * group_size + item.get_local_id(0);

  auto seeds = philox_unpack(philox_args);

  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;

  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int global_idx;
#pragma unroll
  for (int i = 0; i < numel; i += global_size) {
    global_idx = i + idx;
    auto in_offsets = inp_calc.get(global_idx);
    auto out_offsets = out_calc.get(global_idx);
    output_data[out_offsets[0]] =
        f(state, input_data_1[in_offsets[0]], input_data_2[in_offsets[1]]);
  }
}

template <typename func_t>
void distribution_binary_kernel(
    TensorIterator& iter,
    PhiloxState philox_args,
    const func_t& f) {
  static_assert(
      std::is_same<
          typename function_traits<func_t>::template arg<0>::type,
          randStatePhilox4_32_10_t&>::value,
      "the first argument of functor must be randStatePhilox4_32_10_t");
  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;
  using output_t = typename function_traits<func_t>::result_type;

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_binary_kernel(sub_iter, philox_args, f);
    }
    return;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  int group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  int num_groups = (numel + group_size - 1) / group_size;
  int hw_max_groups = dpcppMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  output_t* output_data = static_cast<output_t*>(iter.data_ptr(0));
  const input_t_1* input_data_1 =
      static_cast<const input_t_1*>(iter.data_ptr(1));
  const input_t_2* input_data_2 =
      static_cast<const input_t_2*>(iter.data_ptr(2));

  auto& sycl_queue = dpcppGetCurrentQueue();

  if (iter.is_contiguous()) {
    auto input_offset_calculator = TrivialOffsetCalculator<2>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_binary_elementwise_kernel(
            item,
            numel,
            f,
            philox_args,
            output_data,
            input_data_1,
            input_data_2,
            input_offset_calculator,
            output_offset_calculator);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto input_offset_calculator = make_input_offset_calculator<2>(iter);
    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_binary_elementwise_kernel(
            item,
            numel,
            f,
            philox_args,
            output_data,
            input_data_1,
            input_data_2,
            input_offset_calculator,
            output_offset_calculator);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    size_t engine_calls,
    typename RNG,
    typename transform_t>
void uniform_and_transform(
    TensorIteratorBase& iter,
    RNG gen,
    transform_t transform) {
  // Distribution backbone only handle two accumulate type.
  if (std::is_same<scalar_t, double>::value) {
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls / 2>(
        iter,
        gen,
        [](randStatePhilox4_32_10_t* state) {
          return rand_uniform2_double(state);
        },
        transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls>(
        iter,
        gen,
        [](randStatePhilox4_32_10_t* state) { return rand_uniform4(state); },
        transform);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    size_t engine_calls,
    typename RNG,
    typename transform_t>
void normal_and_transform(
    TensorIteratorBase& iter,
    RNG gen,
    transform_t transform) {
  if (std::is_same<scalar_t, double>::value) {
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls / 2>(
        iter,
        gen,
        [](randStatePhilox4_32_10_t* state) {
          return rand_normal2_double(state);
        },
        transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls>(
        iter,
        gen,
        [](randStatePhilox4_32_10_t* state) { return rand_normal4(state); },
        transform);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
