#pragma once

#include <runtime/Utils.h>
#include "Random.h"
#include "Loops.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename...>
class exponential_sycl_ker {};
template<typename scalar_t, typename accscalar_t, typename dist_t, typename transform_t>
void distribution_elementwise_grid_stride_kernel(at::TensorIterator& iter,
                                                 int numel,
                                                 std::pair<uint64_t, uint64_t> seeds,
                                                 const dist_t dist_func,
                                                 const transform_t transform_func) {

  auto &sycl_queue = dpcppGetCurrentQueue();

  auto offset_calc = make_offset_calculator<1>(iter);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id)  {
      size_t sample_id = item_id.get_id(0);
      auto out_ptr = out_data;
      RandomState<Philox4_32_10> state(seeds.first, sample_id, seeds.second);
      auto rand = dist_func(&state);
      accscalar_t r = ScalarConvert<scalar_t, accscalar_t>::to(rand);
      scalar_t ret = transform_func(r);
      auto offset = offset_calc.get(sample_id)[0];
      scalar_t* out = (scalar_t*)(out_ptr + offset);
      *out = ret;
    };

    cgh.parallel_for<exponential_sycl_ker<scalar_t, accscalar_t, dist_t, transform_t>>(
      DPCPP::range<1>(numel),
      kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

template<typename scalar_t,
  typename accscalar_t,
  typename dist_t,
  typename transform_t>
void distribution_nullary_kernel(at::TensorIterator& iter,
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
      distribution_nullary_kernel<scalar_t, accscalar_t>(sub_iter,
                                                         gen,
                                                         dist_func,
                                                         transform_func);
    }
    return;
  }

  distribution_elementwise_grid_stride_kernel<scalar_t, accscalar_t>(
    iter,
    numel,
    rng_engine_inputs,
    dist_func,
    transform_func);
}

}
}
