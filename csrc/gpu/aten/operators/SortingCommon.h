#pragma once

#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>

#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ApplyUtils.h"

// Maximum size per grid dimension that we assume (compute capability >= 2.0)

using namespace at;

template <typename scalar_t, typename index_t, typename Launcher>
void run_launcher(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    Launcher l) {
  auto self_info = xpu::dpcpp::detail::getTensorInfo<scalar_t, index_t>(self);
  auto values_info =
      xpu::dpcpp::detail::getTensorInfo<scalar_t, index_t>(values);
  auto indices_info =
      xpu::dpcpp::detail::getTensorInfo<int64_t, index_t>(indices);

  int64_t slice_size = self.size(dim);
  /* We use these structures solely to find the offset to */
  /* each slice we are operating on */
  self_info.reduceDim(dim);
  values_info.reduceDim(dim);
  indices_info.reduceDim(dim);

  /* Collapse all other dims */
  int collapse_self_dim = self_info.collapseDims(dim);
  int collapse_values_dim = values_info.collapseDims(dim);
  int collapse_indices_dim = indices_info.collapseDims(dim);

  int64_t num_slices = 1;
  for (int i = 0; i < self_info.dims; ++i) {
    num_slices *= self_info.sizes[i];
  }

  /* This is used as a template parameter to calculate indices. */
  /* We only specialize it if all collapsed dim sizes are the */
  /* same; otherwise, we use -1 which is the specialization */
  /* parameter for arbitrary dimensions */
  int all_dims = self_info.dims;
  if (values_info.dims != all_dims || indices_info.dims != all_dims) {
    all_dims = -1;
  }

  if (all_dims == 1) {
    l.template launch<scalar_t, index_t, 1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 2) {
    l.template launch<scalar_t, index_t, 2>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 3) {
    l.template launch<scalar_t, index_t, 3>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else {
    l.template launch<scalar_t, index_t, -1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  }
}

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
  enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
  enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

template <typename T, int STEPS>
inline void SubgroupScan(
    sycl::sub_group& sg,
    int sgid,
    const T input,
    T& inclusive_sum,
    T& exclusive_sum) {
  inclusive_sum = input;
#pragma unroll
  for (int i = 0, offset = 1; i < STEPS; ++i, offset <<= 1) {
    T temp = sycl::shift_group_right(sg, inclusive_sum, offset);
    if (sgid >= offset)
      inclusive_sum += temp;
  }
  exclusive_sum = inclusive_sum - input;
}

template <
    typename Type,
    int COUNTER_LANES,
    int WORK_ITEMS,
    int SUBGROUP_SIZE,
    bool EXCLUSIVE = true>
inline Type GroupSum(Type* storage, sycl::nd_item<1>& item_id) {
  static_assert(
      WORK_ITEMS % SUBGROUP_SIZE == 0,
      "WORK_ITEMS should be n * SUBGROUP_SIZE. (n = 1, 2, 3, ...)");

  const int NUM_SUBGROUPS = WORK_ITEMS / SUBGROUP_SIZE;
  const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;

  int lid = item_id.get_local_linear_id();
  auto sg = item_id.get_sub_group();

  int subgroup_local_id = sg.get_local_id()[0];
  int subgroup_id = sg.get_group_id()[0];
  int lane_temp_values[COUNTER_LANES];

  // Read input lane sum
  auto storage_lanes = storage + lid * COUNTER_LANES;
  Type lane_all_sum = 0;

  if (EXCLUSIVE) {
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lane_temp_values[lane] = lane_all_sum;
      lane_all_sum += storage_lanes[lane];
    }
  } else {
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lane_all_sum += storage_lanes[lane];
      lane_temp_values[lane] = lane_all_sum;
    }
  }

  // Get subgroup level exclusive sum
  Type subgroup_inclusive_sum, subgroup_exclusive_sum;
  SubgroupScan<Type, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      lane_all_sum,
      subgroup_inclusive_sum,
      subgroup_exclusive_sum);
  item_id.barrier(dpcpp_local_fence);

  // Write to storage
  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    storage[subgroup_id] = subgroup_inclusive_sum;
  item_id.barrier(dpcpp_local_fence);

  // Get block prefix
  Type block_all_sum = 0, block_exclusive_sum;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      block_exclusive_sum = block_all_sum;
    block_all_sum += storage[i];
  }
  item_id.barrier(dpcpp_local_fence);

  // Write to storage
  subgroup_exclusive_sum += block_exclusive_sum;
#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    storage_lanes[lane] = subgroup_exclusive_sum + lane_temp_values[lane];
  }
  item_id.barrier(dpcpp_local_fence);

  return block_all_sum;
}

template <typename Type, int COUNTER_LANES, int WORK_ITEMS, int SUBGROUP_SIZE>
inline Type GroupExclusiveSum(Type* slm_storage, sycl::nd_item<1>& item_id) {
  return GroupSum<Type, COUNTER_LANES, WORK_ITEMS, SUBGROUP_SIZE, true>(
      slm_storage, item_id);
}

template <typename Type, int COUNTER_LANES, int WORK_ITEMS, int SUBGROUP_SIZE>
inline Type GroupInclusiveSum(Type* slm_storage, sycl::nd_item<1>& item_id) {
  return GroupSum<Type, COUNTER_LANES, WORK_ITEMS, SUBGROUP_SIZE, false>(
      slm_storage, item_id);
}
