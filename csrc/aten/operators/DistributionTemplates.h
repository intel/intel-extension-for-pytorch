#pragma once

#include <ATen/core/PhiloxRNGEngine.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "MemoryAccess.h"
#include "RandomEngine.h"
#include "comm/Numerics.h"

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

template <
    typename scalar_t,
    typename accscalar_t,
    int vec_size,
    typename dist_t,
    typename transform_t,
    typename item_t>
inline void distribution_vectorize_kernel(
    item_t& item,
    char* out,
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

  int full_tile_work_size = group_size * num_groups * vec_size;
  int rounded_size =
      ((numel - 1) / full_tile_work_size + 1) * full_tile_work_size;
  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += full_tile_work_size) {
    auto rand = dist_func(&state);
    auto offset = linear_index - idx + idx * vec_size;
    auto remaining = numel - offset;
    if (remaining < vec_size) {
      scalar_t* to_ = reinterpret_cast<scalar_t*>(out);
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        int li = offset + i;
        if (li < numel) {
          to_[li] = transform_func(static_cast<accscalar_t>((&rand.x)[i]));
        }
      }
    } else {
      using vec_t = native::Memory::aligned_vector_loop<scalar_t, vec_size>;
      vec_t* to_ =
          reinterpret_cast<vec_t*>(reinterpret_cast<scalar_t*>(out) + offset);
      vec_t v;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        v.val[i] = transform_func(static_cast<accscalar_t>((&rand.x)[i]));
      *to_ = v;
    }
    item.barrier(dpcpp_local_fence);
  }
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

  int group_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  int num_groups = (numel + group_size - 1) / group_size;
  int hw_max_groups = dpcppMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  uint64_t counter_offset =
      ((numel - 1) / (group_size * num_groups * PHILOX_ENGINE_CALLS) + 1) *
      PHILOX_ENGINE_CALLS;

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

  if ((sizeof(scalar_t) < sizeof(float)) && iter.is_contiguous() &&
      ((reinterpret_cast<uint64_t>(out_data) %
        std::alignment_of<
            native::Memory::aligned_vector_loop<scalar_t, unroll_factor>>::
            value) == 0)) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        distribution_vectorize_kernel<scalar_t, accscalar_t, unroll_factor>(
            item,
            out_data,
            numel,
            PhiloxState(
                std::get<0>(rng_engine_inputs), std::get<1>(rng_engine_inputs)),
            dist_func,
            transform_func);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else if (iter.is_trivial_1d()) {
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
