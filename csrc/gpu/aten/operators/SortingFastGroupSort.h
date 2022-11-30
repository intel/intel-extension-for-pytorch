#pragma once

#include <ATen/ATen.h>
#include <utils/DPCPP.h>

#include "comm/General.h"
#include "comm/KeyTraits.h"
#include "comm/TensorOptions.h"

#include "SortingDeviceMerge.h"
#include "SortingRadixProcesser.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <
    typename KeyT,
    typename ValueT,
    int GROUP_ITEMS,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT = ValueT,
    int SUBGROUP_SIZE>
inline void fast_group_radix_sort_impl_(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    int nsegments,
    int nsort) {
  constexpr int KEYS_PER_ITEM = 4;
  using SortMethod = GroupRadixProcesser<
      KeyT,
      GROUP_ITEMS,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      PrivateValueT>;
  using KeyTraitsT = typename SortMethod::KeyTraitsT;
  int tiles = (nsort + SortMethod::PROCESSING_LENGTH - 1) /
      SortMethod::PROCESSING_LENGTH;
  auto& q = dpcppGetCurrentQueue();

  constexpr uint64_t KEY_TRAITS_TYPE_MASK = 1l
      << ((sizeof(KeyTraitsT) << 3) - 1);
  KeyTraitsT padding_key;
  if (IS_DESCENDING) {
    padding_key = 0;
  } else {
    padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
    padding_key = padding_key ^ (padding_key - 1);
  }

  auto cgf = DPCPP_Q_CGF(h) {
    auto slm = dpcpp_local_acc_t<unsigned char>(
        SortMethod::GetSharedLocalMemorySize(), h);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto gid = item.get_group_linear_id();
      auto lid = item.get_local_id(0);
      auto slice = gid % nsegments;
      auto tid = gid / nsegments;
      int remain_length = nsort - tid * SortMethod::PROCESSING_LENGTH;
      remain_length = remain_length > SortMethod::PROCESSING_LENGTH
          ? SortMethod::PROCESSING_LENGTH
          : remain_length;

      auto offset_s = slice * nsort + tid * SortMethod::PROCESSING_LENGTH;
      auto key_in_begin = key_in + offset_s;
      auto value_in_begin = value_in + offset_s;
      auto key_out_begin = key_out + offset_s;
      auto value_out_begin = value_out + offset_s;

      auto method = SortMethod(item, slm);

      KeyT keys[SortMethod::REG_LEN];
      PrivateValueT values[SortMethod::REG_LEN];

      KeyTraitsT(&ukeys)[KEYS_PER_ITEM] =
          reinterpret_cast<KeyTraitsT(&)[KEYS_PER_ITEM]>(keys);

#pragma unroll
      for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
        int offset = lid * KEYS_PER_ITEM + ITEM;
        if (offset < remain_length) {
          ukeys[ITEM] = KeyTraits<KeyT>::convert(key_in_begin[offset]);
          values[ITEM] = USE_INDICES
              ? tid * SortMethod::PROCESSING_LENGTH + offset
              : value_in_begin[offset];
        } else {
          ukeys[ITEM] = padding_key;
        }
      }

      method.sort_group(keys, values, 0, sizeof(KeyT) * 8);

#pragma unroll
      for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
        int offset = lid * KEYS_PER_ITEM + ITEM;
        if (offset < remain_length) {
          key_out_begin[offset] = KeyTraits<KeyT>::deconvert(ukeys[ITEM]);
          value_out_begin[offset] = values[ITEM];
        }
      }
    };
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(tiles * nsegments * GROUP_ITEMS),
            sycl::range<1>(GROUP_ITEMS)),
        kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

template <
    typename KeyT,
    typename ValueT,
    int GROUP_ITEMS,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT = ValueT>
inline void fast_group_radix_sort_impl(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    int nsegments,
    int nsort) {
  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  switch (dev_prop->subgroup_sizes[0] * 2) {
    // TODO: Fixed subgroup size is used currently for performance consideration
    // however, runtime acquisition is better for scalability
    case 32:
      fast_group_radix_sort_impl_<
          KeyT,
          ValueT,
          GROUP_ITEMS,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          32>(key_in, key_out, value_in, value_out, nsegments, nsort);
      break;
    default:
      fast_group_radix_sort_impl_<
          KeyT,
          ValueT,
          GROUP_ITEMS,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          16>(key_in, key_out, value_in, value_out, nsegments, nsort);
      break;
  }
}

inline uint64_t radix_sort_last_power2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

} // namespace impl

inline std::tuple<int, int, int> get_sorting_workload() {
  constexpr int ITEM_WORK_SIZE = 4;
  constexpr int MERGE_CHUNK_BOUND = 4;
  auto q = dpcppGetDeviceIdOfCurrentQueue();
  auto* dev_prop = dpcppGetDeviceProperties(q);
  auto max_group_size = dpcppMaxWorkGroupSize(q);
  switch (dev_prop->subgroup_sizes[0] * 2) {
    case 32:
      return std::make_tuple(max_group_size, ITEM_WORK_SIZE, MERGE_CHUNK_BOUND);
    default:
      // Reduce local memory pressure
      return std::make_tuple(max_group_size / 2, ITEM_WORK_SIZE, 1);
  }
  return std::make_tuple(0, 0, 0);
}

inline int get_fast_group_sort_bound() {
  auto wl = get_sorting_workload();
  auto group_size = std::get<0>(wl);
  auto item_work_size = std::get<1>(wl);
  auto merge_chunk_bound = std::get<2>(wl);
  return group_size * item_work_size * merge_chunk_bound;
}

template <
    typename KeyT,
    typename ValueT,
    typename PrivateValueT = ValueT,
    bool USE_INDICES = false>
inline void fast_group_sort_pairs(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const int nsegments,
    const int nsort,
    const bool descending) {
  constexpr int ITEM_WORK_SIZE = 4;
  int max_group_sz = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());

  int nsort_th = max_group_sz * ITEM_WORK_SIZE;
  int tiles = (nsort + nsort_th - 1) / nsort_th;

#define GROUP_RADIX_SORT_IMPL(PADDED_NSORT)                        \
  {                                                                \
    if (!descending)                                               \
      impl::fast_group_radix_sort_impl<                            \
          KeyT,                                                    \
          ValueT,                                                  \
          PADDED_NSORT / ITEM_WORK_SIZE,                           \
          false,                                                   \
          USE_INDICES,                                             \
          PrivateValueT>(                                          \
          key_in, key_out, value_in, value_out, nsegments, nsort); \
    else                                                           \
      impl::fast_group_radix_sort_impl<                            \
          KeyT,                                                    \
          ValueT,                                                  \
          PADDED_NSORT / ITEM_WORK_SIZE,                           \
          true,                                                    \
          USE_INDICES,                                             \
          PrivateValueT>(                                          \
          key_in, key_out, value_in, value_out, nsegments, nsort); \
  }
  if (nsort <= nsort_th) {
    switch (impl::radix_sort_last_power2(nsort)) {
      case 4096: // max_group_sz is 1024
        GROUP_RADIX_SORT_IMPL(4096);
        break;
      case 2048:
        GROUP_RADIX_SORT_IMPL(2048);
        break;
      case 1024:
        GROUP_RADIX_SORT_IMPL(1024);
        break;
      case 512:
        GROUP_RADIX_SORT_IMPL(512);
        break;
      default:
        GROUP_RADIX_SORT_IMPL(256);
        break;
    }
  } else {
    switch (max_group_sz) {
      case 1024:
        GROUP_RADIX_SORT_IMPL(1024 * ITEM_WORK_SIZE);
        break;
      case 512:
        GROUP_RADIX_SORT_IMPL(512 * ITEM_WORK_SIZE);
        break;
      default:
        GROUP_RADIX_SORT_IMPL(256 * ITEM_WORK_SIZE);
        break;
    }
  }
#undef GROUP_RADIX_SORT_IMPL
  if (tiles > 1) {
    auto sorting_tmp_k = at::empty({nsegments * nsort}, map_options<KeyT>());
    auto sorting_tmp_v = at::empty({nsegments * nsort}, map_options<ValueT>());
    auto merge_key_temp = (KeyT*)sorting_tmp_k.data_ptr();
    auto merge_value_temp = (ValueT*)sorting_tmp_v.data_ptr();
    if (descending) {
      segmented_device_merge(
          key_out,
          merge_key_temp,
          value_out,
          merge_value_temp,
          std::greater<>(),
          nsegments,
          nsort,
          nsort_th);
    } else {
      segmented_device_merge(
          key_out,
          merge_key_temp,
          value_out,
          merge_value_temp,
          std::less<>(),
          nsegments,
          nsort,
          nsort_th);
    }
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
