#pragma once

#include <runtime/Utils.h>
#include "Loops.h"
#include "Random.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename scalar_t,
    typename accscalar_t,
    typename dist_t,
    typename transform_t>
void distribution_elementwise_grid_stride_kernel(
    at::TensorIterator& iter,
    int numel,
    std::pair<uint64_t, uint64_t> seeds,
    const dist_t dist_func,
    const transform_t transform_func) {
  constexpr int unroll_factor = sizeof(accscalar_t) <= 4 ? 4 : 2;
  auto& sycl_queue = dpcppGetCurrentQueue();
  int group_items = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int group_work_size = group_items * unroll_factor;
  int num_groups = (numel + group_work_size - 1) / group_work_size;
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto out_data = (char*)iter.data_ptr(0);
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
        int gid = item.get_group(0);
        int tid = item.get_local_id(0);
        RandomState<Philox4_32_10> state(
            seeds.first, gid * group_items + tid, seeds.second);
        int sample_id = gid * group_work_size + tid;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) {
          if (sample_id >= numel)
            return;
          auto rand = dist_func(&state);
          accscalar_t r = ScalarConvert<scalar_t, accscalar_t>::to(rand);
          scalar_t ret = transform_func(r);
          auto offset = sample_id * stride0;
          scalar_t* out = (scalar_t*)(out_data + offset);
          *out = ret;
          sample_id += group_items;
        }
      };
      cgh.parallel_for(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(num_groups * group_items),
              DPCPP::range<1>(group_items)),
          kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto out_data = (char*)iter.data_ptr(0);
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
        int gid = item.get_group(0);
        int tid = item.get_local_id(0);
        RandomState<Philox4_32_10> state(
            seeds.first, gid * group_items + tid, seeds.second);
        int sample_id = gid * group_work_size + tid;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) {
          if (sample_id >= numel)
            return;
          auto rand = dist_func(&state);
          accscalar_t r = ScalarConvert<scalar_t, accscalar_t>::to(rand);
          scalar_t ret = transform_func(r);
          auto offset = offset_calc.get(sample_id)[0];
          scalar_t* out = (scalar_t*)(out_data + offset);
          *out = ret;
          sample_id += group_items;
        }
      };
      cgh.parallel_for(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(num_groups * group_items),
              DPCPP::range<1>(group_items)),
          kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename dist_t,
    typename transform_t>
void distribution_nullary_kernel(
    at::TensorIterator& iter,
    xpu::dpcpp::DPCPPGeneratorImpl* gen,
    const dist_t& dist_func,
    const transform_t transform_func) {
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t>(
          sub_iter, gen, dist_func, transform_func);
    }
    return;
  }

  distribution_elementwise_grid_stride_kernel<scalar_t, accscalar_t>(
      iter, numel, rng_engine_inputs, dist_func, transform_func);
}

} // namespace AtenIpexTypeXPU
} // namespace at
