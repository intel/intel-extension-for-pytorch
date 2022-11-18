#pragma once

#include <ATen/ATen.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include "comm/Numerics.h"

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

#define CALL_MIN_2(X, Y) ((X) < (Y) ? (X) : (Y))

template <typename KeyT, typename ValueT, typename Compare>
inline void device_merge_chunk(
    KeyT* key,
    KeyT* key_out,
    ValueT* value,
    ValueT* value_out,
    int offset,
    int chunk,
    int start_1, // relative position
    int end_1,
    int end_2,
    Compare comp) {
  int start_2 = end_1;
  int local_start_1 = CALL_MIN_2(offset + start_1, end_1);
  int local_end_1 = CALL_MIN_2(local_start_1 + chunk, end_1);
  int local_start_2 = CALL_MIN_2(offset + start_2, end_2);
  int local_end_2 = CALL_MIN_2(local_start_2 + chunk, end_2);
  int local_size_1 = local_end_1 - local_start_1;
  int local_size_2 = local_end_2 - local_start_2;
  // Process 1st sequence
  if (local_start_1 < local_end_1) {
    // Reduce the range for searching within the 2nd sequence and handle bound
    // items find left border in 2nd sequence
    auto local_l_item_1_key = key[local_start_1];
    auto local_l_item_1_value = value[local_start_1];
    int l_search_bound_2 =
        std::lower_bound(key + start_2, key + end_2, local_l_item_1_key, comp) -
        key;
    int l_shift_1 = local_start_1 - start_1;
    int l_shift_2 = l_search_bound_2 - start_2;
    key_out[start_1 + l_shift_1 + l_shift_2] = local_l_item_1_key;
    value_out[start_1 + l_shift_1 + l_shift_2] = local_l_item_1_value;
    int r_search_bound_2;
    // find right border in 2nd sequence
    if (local_size_1 > 1) {
      auto local_r_item_1_key = key[local_end_1 - 1];
      auto local_r_item_1_value = value[local_end_1 - 1];
      r_search_bound_2 =
          std::lower_bound(
              key + l_search_bound_2, key + end_2, local_r_item_1_key, comp) -
          key;
      int r_shift_1 = local_end_1 - 1 - start_1;
      int r_shift_2 = r_search_bound_2 - start_2;
      key_out[start_1 + r_shift_1 + r_shift_2] = local_r_item_1_key;
      value_out[start_1 + r_shift_1 + r_shift_2] = local_r_item_1_value;
    }
    // Handle intermediate items
    for (int idx = local_start_1 + 1; idx < local_end_1 - 1; idx++) {
      auto intermediate_item_1_key = key[idx];
      auto intermediate_item_1_value = value[idx];
      l_search_bound_2 = std::lower_bound(
                             key + l_search_bound_2,
                             key + r_search_bound_2,
                             intermediate_item_1_key,
                             comp) -
          key;
      int shift_1 = idx - start_1;
      int shift_2 = l_search_bound_2 - start_2;
      key_out[start_1 + shift_1 + shift_2] = intermediate_item_1_key;
      value_out[start_1 + shift_1 + shift_2] = intermediate_item_1_value;
    }
  }
  // Process 2nd sequence
  if (local_start_2 < local_end_2) {
    // Reduce the range for searching within the 1st sequence and handle bound
    // items find left border in 1st sequence
    auto local_l_item_2_key = key[local_start_2];
    auto local_l_item_2_value = value[local_start_2];
    int l_search_bound_1 =
        std::upper_bound(key + start_1, key + end_1, local_l_item_2_key, comp) -
        key;
    int l_shift_1 = l_search_bound_1 - start_1;
    int l_shift_2 = local_start_2 - start_2;
    key_out[start_1 + l_shift_1 + l_shift_2] = local_l_item_2_key;
    value_out[start_1 + l_shift_1 + l_shift_2] = local_l_item_2_value;
    int r_search_bound_1;
    // find right border in 1st sequence
    if (local_size_2 > 1) {
      auto local_r_item_2_key = key[local_end_2 - 1];
      auto local_r_item_2_value = value[local_end_2 - 1];
      r_search_bound_1 =
          std::upper_bound(
              key + l_search_bound_1, key + end_1, local_r_item_2_key, comp) -
          key;
      int r_shift_1 = r_search_bound_1 - start_1;
      int r_shift_2 = local_end_2 - 1 - start_2;
      key_out[start_1 + r_shift_1 + r_shift_2] = local_r_item_2_key;
      value_out[start_1 + r_shift_1 + r_shift_2] = local_r_item_2_value;
    }
    // Handle intermediate items
    for (int idx = local_start_2 + 1; idx < local_end_2 - 1; idx++) {
      auto intermediate_item_2_key = key[idx];
      auto intermediate_item_2_value = value[idx];
      l_search_bound_1 = std::upper_bound(
                             key + l_search_bound_1,
                             key + r_search_bound_1,
                             intermediate_item_2_key,
                             comp) -
          key;
      int shift_1 = l_search_bound_1 - start_1;
      int shift_2 = idx - start_2;
      key_out[start_1 + shift_1 + shift_2] = intermediate_item_2_key;
      value_out[start_1 + shift_1 + shift_2] = intermediate_item_2_value;
    }
  }
}

template <typename KeyT, typename ValueT, typename Compare, typename ItemT>
inline void device_merge_full(
    KeyT* key,
    KeyT* key_temp,
    ValueT* value,
    ValueT* value_temp,
    int chunk_size,
    int size_have_sorted,
    int length,
    Compare comp,
    ItemT& item) {
  const auto chunk = chunk_size;
  auto lid = item.get_local_linear_id();
  bool data_in_temp = false;
  auto sorted_size = size_have_sorted / chunk_size;
  while (sorted_size * chunk < length) {
    const auto start_1 =
        CALL_MIN_2(2 * sorted_size * chunk * (lid / sorted_size), length);
    auto end_1 = CALL_MIN_2(start_1 + sorted_size * chunk, length);
    auto start_2 = end_1;
    auto end_2 = CALL_MIN_2(start_2 + sorted_size * chunk, length);
    auto offset = chunk * (lid % sorted_size);
    if (!data_in_temp) {
      device_merge_chunk(
          key,
          key_temp,
          value,
          value_temp,
          offset,
          chunk,
          start_1,
          end_1,
          end_2,
          comp);
    } else {
      device_merge_chunk(
          key_temp,
          key,
          value_temp,
          value,
          offset,
          chunk,
          start_1,
          end_1,
          end_2,
          comp);
    }
    item.barrier(dpcpp_local_fence);
    data_in_temp = !data_in_temp;
    sorted_size *= 2;
  }

  if (data_in_temp) {
    for (int idx = item.get_local_id(0); idx < length;
         idx += item.get_local_range(0)) {
      key[idx] = key_temp[idx];
      value[idx] = value_temp[idx];
    }
  }
}

template <typename KeyT, typename ValueT, typename Compare>
inline void segmented_device_merge(
    KeyT* key_ptr,
    KeyT* key_temp_ptr,
    ValueT* value_ptr,
    ValueT* value_temp_ptr,
    Compare comp,
    int nsegments,
    int nsort,
    int size_have_sorted = 1) {
  int max_group_sz = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int chunk_size =
      (size_have_sorted % 16 == 0) ? 16 : size_have_sorted; // need tune
  int group_sz = (nsort + chunk_size - 1) / chunk_size;
  auto& q = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(h) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto offset = item.get_group_linear_id() * nsort;
      auto key_slice_start = key_ptr + offset;
      auto value_slice_start = value_ptr + offset;
      auto key_temp_slice_start = key_temp_ptr + offset;
      auto value_temp_slice_start = value_temp_ptr + offset;
      device_merge_full(
          key_slice_start,
          key_temp_slice_start,
          value_slice_start,
          value_temp_slice_start,
          chunk_size,
          size_have_sorted,
          nsort,
          comp,
          item);
    };
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(nsegments * group_sz), sycl::range<1>(group_sz)),
        kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

#undef CALL_MIN_2

} // namespace AtenIpexTypeXPU
} // namespace at
