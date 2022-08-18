#pragma once

#include <utils/DPCPP.h>
#include "SortingCommon.h"
#include "comm/General.h"
#include "comm/KeyTraits.h"

#include "SortingDeviceMerge.h"

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename KeyT,
    int GROUP_THREADS,
    int SUBGROUP_SIZE,
    int KEYS_PER_THREAD,
    bool IS_DESCENDING = false,
    typename ValueT = NullType, // NullType is set to be default.
    bool USE_INDICES = false,
    typename PrivateValueT = ValueT,
    typename DigitT = uint16_t, // Counters' datatype. The value is less than
                                // GROUP_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t,
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS = 4>
class GroupRadixSort {
 public:
  static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
  static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
  static_assert(
      ((1 << (sizeof(DigitT) << 3)) - 1) >= (GROUP_THREADS * KEYS_PER_THREAD),
      "");
  using KeyTraitsT = typename KeyTraits<KeyT>::Type;

  enum {
    SORT_LEN = GROUP_THREADS * KEYS_PER_THREAD,
    REG_LEN = KEYS_PER_THREAD,
    RADIX_BUCKETS = 1 << RADIX_BITS,
    KEYS_ONLY = std::is_same<ValueT, NullType>::value,
    PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
    COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
    LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
    PACKED_SCAN_SIZE = COUNTER_LANES * GROUP_THREADS * sizeof(CounterT),
    DIGIT_BITS = sizeof(DigitT) << 3,
    DIGIT_MASK = (1 << DIGIT_BITS) - 1,
    KEY_TRAITS_TYPE_MASK = 1l << ((sizeof(KeyTraitsT) << 3) - 1),
  };

 private:
  sycl::nd_item<1>& item_id;
  int lid;
  char* local_storage;
  int tiles;
  int tid;

 public:
  static int GetSharedLocalStorageSize() {
    const int KV_TYPE_SIZE = KEYS_ONLY
        ? sizeof(KeyT)
        : std::max(sizeof(KeyT), sizeof(PrivateValueT));
    return std::max(
        GROUP_THREADS * KEYS_PER_THREAD * KV_TYPE_SIZE, (int)PACKED_SCAN_SIZE);
  }

  inline GroupRadixSort(
      sycl::nd_item<1>& item_id,
      char* local_storage,
      int tid = 0)
      : item_id(item_id), local_storage(local_storage), tid(tid) {
    lid = item_id.get_local_linear_id();
  }

  inline void SortBlocked(
      KeyTraitsT (&pkey)[KEYS_PER_THREAD],
      PrivateValueT (&pvalue)[KEYS_PER_THREAD],
      int begin_bit,
      int end_bit) {
    while (true) {
#define RADIX_NUMERIC_MIN(A, B) (((A) > (B)) ? (B) : (A))
      auto pass_bits = RADIX_NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
#undef RADIX_NUMERIC_MIN
      int rank[KEYS_PER_THREAD];
      RankKeys(pkey, rank, begin_bit, pass_bits);
      Exchange<KeyTraitsT>(pkey, rank);
      if (!KEYS_ONLY)
        Exchange<PrivateValueT>(pvalue, rank);
      begin_bit += RADIX_BITS;
      if (begin_bit >= end_bit)
        break;
    }
  }

  inline void ReadKeyFromGlobal(
      KeyTraitsT (&pkey)[KEYS_PER_THREAD],
      const KeyT* key,
      int length,
      int stride = 1) {
    KeyTraitsT PADDING_KEY;
    if (IS_DESCENDING) {
      PADDING_KEY = 0;
    } else {
      PADDING_KEY = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
      PADDING_KEY = PADDING_KEY ^ (PADDING_KEY - 1);
    }
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid * KEYS_PER_THREAD + ITEM;
      pkey[ITEM] = (offset < length)
          ? KeyTraits<KeyT>::convert(key[offset * stride])
          : PADDING_KEY;
    }
  }

  inline void ReadValueFromGlobal(
      PrivateValueT (&pvalue)[KEYS_PER_THREAD],
      const ValueT* value,
      int length,
      int stride = 1) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid * KEYS_PER_THREAD + ITEM;
      if (!USE_INDICES) {
        if (offset < length)
          pvalue[ITEM] = value[offset * stride];
      } else {
        pvalue[ITEM] = tid * SORT_LEN + offset;
      }
    }
  }

  inline void WriteKeyToGlobal(
      KeyT* key,
      KeyTraitsT (&pkey)[KEYS_PER_THREAD],
      int length,
      int stride = 1) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid * KEYS_PER_THREAD + ITEM;
      if (offset < length)
        key[offset * stride] = KeyTraits<KeyT>::deconvert(pkey[ITEM]);
    }
  }

  inline void WriteValueToGlobal(
      ValueT* value,
      PrivateValueT (&pvalue)[KEYS_PER_THREAD],
      int length,
      int stride = 1) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid * KEYS_PER_THREAD + ITEM;
      if (offset < length)
        value[offset * stride] = pvalue[ITEM];
    }
  }

  template <typename DataT>
  inline void Exchange(
      DataT (&data)[KEYS_PER_THREAD],
      int (&rank)[KEYS_PER_THREAD]) {
    auto local_storage_ = reinterpret_cast<DataT*>(local_storage);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
      local_storage_[rank[ITEM]] = data[ITEM];
    item_id.barrier(dpcpp_local_fence);
    auto local_storage_lid = local_storage_ + lid * KEYS_PER_THREAD;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
      data[ITEM] = local_storage_lid[ITEM];
    item_id.barrier(dpcpp_local_fence);
  }

  inline DigitT ExtractDigit(KeyTraitsT key, int begin, int pass) {
    return ((key >> begin) & ((1 << pass) - 1));
  }

  inline void RankKeys(
      KeyTraitsT (&key)[KEYS_PER_THREAD],
      int (&rank)[KEYS_PER_THREAD],
      int begin_bit,
      int pass_bits) {
    DigitT* digit_counters[KEYS_PER_THREAD];
    DigitT sub_counters[KEYS_PER_THREAD];
    auto scan_storage = reinterpret_cast<CounterT*>(local_storage);
    auto buckets = reinterpret_cast<DigitT(*)[GROUP_THREADS][PACKING_RATIO]>(
        local_storage);

    // Reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM)
      scan_storage[lid * COUNTER_LANES + ITEM] = 0; // fast
    item_id.barrier(dpcpp_local_fence);

    // Bin
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      auto digit = ExtractDigit(key[ITEM], begin_bit, pass_bits);
      auto sub_counter = digit >> LOG_COUNTER_LANES;
      auto counter_lane = digit & (COUNTER_LANES - 1);
      if (IS_DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }
      sub_counters[ITEM] = sub_counter;
      digit_counters[ITEM] = &buckets[counter_lane][lid][sub_counter];
      rank[ITEM] = *digit_counters[ITEM];
      *digit_counters[ITEM] = rank[ITEM] + 1;
    }
    item_id.barrier(dpcpp_local_fence);

    // Exclusive scan
    CounterT temp = GroupExclusiveSum<
        CounterT,
        COUNTER_LANES,
        GROUP_THREADS,
        SUBGROUP_SIZE>(scan_storage, item_id);

    // Decode packing data
    CounterT c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      temp = temp << DIGIT_BITS;
      c += temp;
    }

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      DigitT cc = (c >> (sub_counters[ITEM] * DIGIT_BITS)) & DIGIT_MASK;
      rank[ITEM] += *digit_counters[ITEM] + cc;
    }
    item_id.barrier(dpcpp_local_fence);
  }
};

template <
    typename KeyT,
    typename ValueT,
    typename OffsetCalT,
    int GROUP_THREADS,
    int KEYS_PER_THREAD,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT = ValueT,
    int SUBGROUP_SIZE>
inline void segmented_group_radix_sort_impl_(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    int nsegments,
    int nsort,
    int stride,
    int tiles = 1) {
  using SortMethod = GroupRadixSort<
      KeyT,
      GROUP_THREADS,
      SUBGROUP_SIZE,
      KEYS_PER_THREAD,
      IS_DESCENDING,
      ValueT,
      USE_INDICES,
      PrivateValueT>;
  auto& q = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(h) {
    auto slm =
        dpcpp_local_acc_t<char>(SortMethod::GetSharedLocalStorageSize(), h);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto gid = item.get_group_linear_id();
      auto slice = gid % nsegments;
      auto tid = gid / nsegments;
      int remain_length = nsort - tid * SortMethod::SORT_LEN;
      remain_length = remain_length > SortMethod::SORT_LEN
          ? SortMethod::SORT_LEN
          : remain_length;

      auto offset = f(slice) + tid * SortMethod::SORT_LEN * stride;
      auto key_in_begin = key_in + offset;
      auto value_in_begin = value_in + offset;

      KeyT* key_out_begin;
      ValueT* value_out_begin;
      int stride_out;
      bool force_to_last = tiles > 1;
      if (!force_to_last) {
        key_out_begin = key_out + offset;
        value_out_begin = value_out + offset;
        stride_out = stride;
      } else {
        // transpose to last dim for efficiency
        auto offset_ = slice * nsort + tid * SortMethod::SORT_LEN;
        key_out_begin = key_out + offset_;
        value_out_begin = value_out + offset_;
        stride_out = 1;
      }

      auto method = SortMethod(item, (char*)slm.get_pointer().get(), tid);
      using KeyTraitsT = typename SortMethod::KeyTraitsT;
      KeyTraitsT pkey[SortMethod::REG_LEN];
      PrivateValueT pvalue[SortMethod::REG_LEN];

      method.ReadKeyFromGlobal(pkey, key_in_begin, remain_length, stride);
      method.ReadValueFromGlobal(pvalue, value_in_begin, remain_length, stride);
      method.SortBlocked(pkey, pvalue, 0, KeyTraits<KeyT>::endbit());
      method.WriteKeyToGlobal(key_out_begin, pkey, remain_length, stride_out);
      method.WriteValueToGlobal(
          value_out_begin, pvalue, remain_length, stride_out);
    };
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(tiles * nsegments * GROUP_THREADS),
            sycl::range<1>(GROUP_THREADS)),
        kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

template <
    typename KeyT,
    typename ValueT,
    typename OffsetCalT,
    int GROUP_THREADS,
    int KEYS_PER_THREAD,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT = ValueT>
inline void segmented_group_radix_sort_impl(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    int nsegments,
    int nsort,
    int stride,
    int tiles = 1) {
  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  switch (dev_prop->subgroup_sizes[0] * 2) {
    // TODO: Fixed subgroup size is used currently for performance consideration
    // however, runtime acquisition is better for scalability
    case 32:
      segmented_group_radix_sort_impl_<
          KeyT,
          ValueT,
          OffsetCalT,
          GROUP_THREADS,
          KEYS_PER_THREAD,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          32>(
          key_in,
          key_out,
          value_in,
          value_out,
          f,
          nsegments,
          nsort,
          stride,
          tiles);
      break;
    default:
      segmented_group_radix_sort_impl_<
          KeyT,
          ValueT,
          OffsetCalT,
          GROUP_THREADS,
          KEYS_PER_THREAD,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          16>(
          key_in,
          key_out,
          value_in,
          value_out,
          f,
          nsegments,
          nsort,
          stride,
          tiles);
      break;
  }
}

struct SegmentedGroupRadixSortDesc {
  enum { ITEM_WORK_SIZE = 4 }; // To reduce register work load, set 4 as default

  int nsegments;
  int nsort;
  int stride;
  bool descending;
  bool use_u16_indices;

  int max_group_sz;
  int group_radix_sort_th;

  SegmentedGroupRadixSortDesc() {}
  SegmentedGroupRadixSortDesc(
      const int nsegments,
      const int nsort,
      const int stride,
      const bool descending,
      const bool use_u16_indices = false)
      : nsegments(nsegments),
        nsort(nsort),
        stride(stride),
        descending(descending),
        use_u16_indices(use_u16_indices) {
    max_group_sz = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
    group_radix_sort_th = ITEM_WORK_SIZE * max_group_sz;
  }

  inline bool valid() {
    if (nsort <= group_radix_sort_th)
      return true;
    else
      return nsort <= 16 * max_group_sz; // Currently, device merge is carried
                                         // out within 16 * max_group_sz
  }

  inline bool need_temp() {
    return nsort > group_radix_sort_th;
  }
};

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

template <
    typename KeyT,
    typename ValueT,
    typename PrivateValueT,
    bool USE_INDICES,
    typename OffsetCalT,
    typename SegmentedGroupRadixSortDescT>
inline void segmented_group_radix_sort_kernel(
    SegmentedGroupRadixSortDescT& desc,
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    KeyT* key_temp = nullptr,
    ValueT* value_temp = nullptr) {
  TORCH_CHECK(desc.valid());
  constexpr int ITEM_WORK_SIZE = SegmentedGroupRadixSortDescT::ITEM_WORK_SIZE;
  int nsort_th = desc.group_radix_sort_th;

  auto nsegments = desc.nsegments;
  auto nsort = desc.nsort;
  auto stride = desc.stride;
  auto descending = desc.descending;
  int tiles = (nsort + nsort_th - 1) / nsort_th;

  KeyT *radix_key_out, *merge_key_temp;
  ValueT *radix_value_out, *merge_value_temp;
  if (desc.stride > 1 && tiles > 1) {
    radix_key_out = key_temp;
    merge_key_temp = key_out;
    radix_value_out = value_temp;
    merge_value_temp = value_out;
  } else {
    radix_key_out = key_out;
    merge_key_temp = key_temp;
    radix_value_out = value_out;
    merge_value_temp = value_temp;
  }

#define GROUP_RADIX_SORT_IMPL(PADDED_NSORT) \
  {                                         \
    if (!descending)                        \
      segmented_group_radix_sort_impl<      \
          KeyT,                             \
          ValueT,                           \
          OffsetCalT,                       \
          PADDED_NSORT / ITEM_WORK_SIZE,    \
          ITEM_WORK_SIZE,                   \
          false,                            \
          USE_INDICES,                      \
          PrivateValueT>(                   \
          key_in,                           \
          radix_key_out,                    \
          value_in,                         \
          radix_value_out,                  \
          f,                                \
          nsegments,                        \
          nsort,                            \
          stride,                           \
          tiles);                           \
    else                                    \
      segmented_group_radix_sort_impl<      \
          KeyT,                             \
          ValueT,                           \
          OffsetCalT,                       \
          PADDED_NSORT / ITEM_WORK_SIZE,    \
          ITEM_WORK_SIZE,                   \
          true,                             \
          USE_INDICES,                      \
          PrivateValueT>(                   \
          key_in,                           \
          radix_key_out,                    \
          value_in,                         \
          radix_value_out,                  \
          f,                                \
          nsegments,                        \
          nsort,                            \
          stride,                           \
          tiles);                           \
  }
  if (nsort <= nsort_th) {
    switch (radix_sort_last_power2(nsort)) {
      case 4096:
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
    switch (desc.max_group_sz) {
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
    if (descending) {
      segmented_device_merge(
          radix_key_out,
          merge_key_temp,
          radix_value_out,
          merge_value_temp,
          std::greater<>(),
          nsegments,
          nsort,
          desc.group_radix_sort_th);
    } else {
      segmented_device_merge(
          radix_key_out,
          merge_key_temp,
          radix_value_out,
          merge_value_temp,
          std::less<>(),
          nsegments,
          nsort,
          desc.group_radix_sort_th);
    }
    if (radix_key_out != key_out) {
      // reorder if need
      auto& q = dpcppGetCurrentQueue();
      auto cgf = DPCPP_Q_CGF(h) {
        auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
          int gid = item.get_group(0);
          auto offset_in = gid * desc.nsort;
          auto offset_out = f(gid);
          auto key_in_begin = radix_key_out + offset_in;
          auto value_in_begin = radix_value_out + offset_in;
          auto key_out_begin = key_out + offset_out;
          auto value_out_begin = value_out + offset_out;
          for (int ni = item.get_local_id(0); ni < desc.nsort;
               ni += item.get_local_range(0)) {
            key_out_begin[ni * desc.stride] = key_in_begin[ni];
            value_out_begin[ni * desc.stride] = value_in_begin[ni];
          }
        };
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(desc.nsegments * desc.max_group_sz),
                sycl::range<1>(desc.max_group_sz)),
            kfn);
      };
      DPCPP_Q_SUBMIT(q, cgf);
    }
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
