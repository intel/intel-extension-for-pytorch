#pragma once

#include <utils/DPCPP.h>
#include <limits>
#include "SortingCommon.h"
#include "comm/General.h"
#include "comm/KeyTraits.h"

namespace at {
namespace AtenIpexTypeXPU {

template <int COUNTER_LANES, int PACKING_RATIO, typename T>
inline void radix_count(
    const T digit,
    T& counter_lane,
    T& sub_counter,
    bool is_descending) {
  constexpr int LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE;
  counter_lane = digit & (COUNTER_LANES - 1);
  sub_counter = digit >> LOG_COUNTER_LANES;
  if (is_descending) {
    counter_lane = COUNTER_LANES - 1 - counter_lane;
    sub_counter = PACKING_RATIO - 1 - sub_counter;
  }
}

template <int GROUP_THREADS, typename DigitT, typename CounterT, int RADIX_BITS>
int buckets_slm_bytes() {
  const int RADIX_BUCKETS = 1 << RADIX_BITS;
  const int PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT);
  const int COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO;
  return COUNTER_LANES * GROUP_THREADS * sizeof(CounterT);
}

template <
    typename KeyT,
    typename ValueT,
    typename OffsetCalT,
    int GROUP_THREADS,
    int SUBGROUP_SIZE,
    typename DigitT,
    typename CounterT,
    int RADIX_BITS>
inline void device_radix_sort_kernel(
    sycl::nd_item<1>& item,
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    KeyT* key_temp,
    ValueT* value_temp,
    int keys_per_thread,
    int nsort,
    int stride,
    void* slm,
    bool is_descending,
    bool use_indices) {
  const int RADIX_BUCKETS = 1 << RADIX_BITS;
  const int PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT);
  const int COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO;
  const int DIGIT_BITS = sizeof(DigitT) << 3;
  const int DIGIT_MASK = (1ul << DIGIT_BITS) - 1;
  using KeyTraitsT = typename KeyTraits<KeyT>::Type;

  auto scan_storage = reinterpret_cast<CounterT*>(slm);
  auto buckets = reinterpret_cast<DigitT(*)[GROUP_THREADS][PACKING_RATIO]>(slm);

  auto lid = item.get_local_id(0);
  auto bx = item.get_group(0);

  KeyT* key_src = key_out;
  KeyT* key_dst = key_temp;
  ValueT* value_src = value_out;
  ValueT* value_dst = value_temp;
  int begin_bit = 0;
  int end_bit = KeyTraits<KeyT>::endbit();

  while (true) {
#define RADIX_NUMERIC_MIN(A, B) (((A) > (B)) ? (B) : (A))
    auto pass_bits = RADIX_NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
#undef RADIX_NUMERIC_MIN

    // Reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM)
      scan_storage[lid * COUNTER_LANES + ITEM] = 0;
    item.barrier(dpcpp_local_fence);

    // Write key digits to buckets
    for (int ITEM = 0; ITEM < keys_per_thread; ++ITEM) {
      int offset = lid * keys_per_thread + ITEM;
      if (offset < nsort) {
        auto k =
            begin_bit == 0 ? key_in[offset * stride] : key_src[offset * stride];
        KeyTraitsT trait = KeyTraits<KeyT>::convert(k);
        DigitT digit = (trait >> begin_bit) & ((1 << pass_bits) - 1);
        DigitT counter_lane, sub_counter;
        radix_count<COUNTER_LANES, PACKING_RATIO>(
            digit, counter_lane, sub_counter, is_descending);
        buckets[counter_lane][lid][sub_counter] += 1;
      }
    }
    item.barrier(dpcpp_local_fence);

    // Exclusive scan
    CounterT temp = GroupExclusiveSum<
        CounterT,
        COUNTER_LANES,
        GROUP_THREADS,
        SUBGROUP_SIZE>(scan_storage, item);

    // Decode packing data
    CounterT c = 0;
    if (PACKING_RATIO != 1) {
#pragma unroll
      for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
        temp = temp << DIGIT_BITS;
        c += temp;
      }
    }

    // Reorder
    for (int ITEM = 0; ITEM < keys_per_thread; ++ITEM) {
      int offset = lid * keys_per_thread + ITEM;
      if (offset < nsort) {
        auto k =
            begin_bit == 0 ? key_in[offset * stride] : key_src[offset * stride];
        KeyTraitsT trait = KeyTraits<KeyT>::convert(k);
        DigitT digit = (trait >> begin_bit) & ((1 << pass_bits) - 1);
        DigitT counter_lane, sub_counter;
        radix_count<COUNTER_LANES, PACKING_RATIO>(
            digit, counter_lane, sub_counter, is_descending);
        int rank;
        if (PACKING_RATIO != 1) {
          DigitT cc = (c >> (sub_counter * DIGIT_BITS)) & DIGIT_MASK;
          rank = buckets[counter_lane][lid][sub_counter] + cc;
        } else {
          rank = buckets[counter_lane][lid][sub_counter];
        }
        // exchange mem
        key_dst[rank * stride] = k;
        if (use_indices && begin_bit == 0) {
          value_dst[rank * stride] = offset;
        } else {
          auto v = begin_bit == 0 ? value_in[offset * stride]
                                  : value_src[offset * stride];
          value_dst[rank * stride] = v;
        }
        buckets[counter_lane][lid][sub_counter] += 1;
      }
    }

    std::swap(key_src, key_dst);
    std::swap(value_src, value_dst);

    item.barrier(dpcpp_local_fence);
    begin_bit += RADIX_BITS;
    if (begin_bit >= end_bit)
      break;
  }
}

template <
    typename KeyT,
    typename ValueT,
    typename OffsetCalT,
    int GROUP_THREADS,
    typename DigitT,
    typename CounterT,
    int RADIX_BITS>
inline void device_radix_sort_impl(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    KeyT* key_temp,
    ValueT* value_temp,
    int nsegments,
    int nsort,
    int stride,
    bool is_descending,
    bool use_indices) {
  int keys_per_thread = (nsort + GROUP_THREADS - 1) / GROUP_THREADS;
  int slm_size =
      buckets_slm_bytes<GROUP_THREADS, DigitT, CounterT, RADIX_BITS>();
#define DISPATCH_SG(SGSZ)                                \
  {                                                      \
    auto& q = dpcppGetCurrentQueue();                    \
    auto cgf = DPCPP_Q_CGF(h) {                          \
      auto slm = dpcpp_local_acc_t<char>(slm_size, h);   \
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)      \
          [[intel::reqd_sub_group_size(SGSZ)]] {         \
        auto slice = item.get_group_linear_id();         \
        auto offset_s = f(slice);                        \
        auto key_in_begin = key_in + offset_s;           \
        auto key_out_begin = key_out + offset_s;         \
        auto key_temp_begin = key_temp + offset_s;       \
        auto value_in_begin = value_in + offset_s;       \
        auto value_out_begin = value_out + offset_s;     \
        auto value_temp_begin = value_temp + offset_s;   \
        device_radix_sort_kernel<                        \
            KeyT,                                        \
            ValueT,                                      \
            OffsetCalT,                                  \
            GROUP_THREADS,                               \
            SGSZ,                                        \
            DigitT,                                      \
            CounterT,                                    \
            RADIX_BITS>(                                 \
            item,                                        \
            key_in_begin,                                \
            key_out_begin,                               \
            value_in_begin,                              \
            value_out_begin,                             \
            f,                                           \
            key_temp_begin,                              \
            value_temp_begin,                            \
            keys_per_thread,                             \
            nsort,                                       \
            stride,                                      \
            (void*)slm.get_pointer().get(),              \
            is_descending,                               \
            use_indices);                                \
      };                                                 \
      h.parallel_for(                                    \
          sycl::nd_range<1>(                             \
              sycl::range<1>(nsegments * GROUP_THREADS), \
              sycl::range<1>(GROUP_THREADS)),            \
          kfn);                                          \
    };                                                   \
    DPCPP_Q_SUBMIT(q, cgf);                              \
  }

  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  switch (dev_prop->subgroup_sizes[0] * 2) {
    case 32:
      DISPATCH_SG(32);
      break;
    default:
      DISPATCH_SG(16);
      break;
  }

#undef DISPATCH_SG
}

struct SegmentedDeviceRadixSortDesc {
  int nsegments;
  int nsort;
  int stride;
  bool descending;
  int use_indices;

  int max_group_sz;

  SegmentedDeviceRadixSortDesc() {}
  SegmentedDeviceRadixSortDesc(
      const int nsegments,
      const int nsort,
      const int stride,
      const bool descending,
      const bool use_indices = false)
      : nsegments(nsegments),
        nsort(nsort),
        stride(stride),
        descending(descending),
        use_indices(use_indices) {
    max_group_sz = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  }

  inline bool valid() {
    return nsort <= std::numeric_limits<int>::max();
  }
};

template <
    typename KeyT,
    typename ValueT,
    typename OffsetCalT,
    typename SegmentedDeviceRadixSortDescT>
inline void segmented_device_radix_sort_kernel(
    SegmentedDeviceRadixSortDescT& desc,
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    const OffsetCalT& f,
    KeyT* key_temp,
    ValueT* value_temp) {
#define DISPATCH_GSZ(GSZ)   \
  {                         \
    device_radix_sort_impl< \
        KeyT,               \
        ValueT,             \
        OffsetCalT,         \
        GSZ,                \
        uint32_t,           \
        uint32_t,           \
        4>(                 \
        key_in,             \
        key_out,            \
        value_in,           \
        value_out,          \
        f,                  \
        key_temp,           \
        value_temp,         \
        desc.nsegments,     \
        desc.nsort,         \
        desc.stride,        \
        desc.descending,    \
        desc.use_indices);  \
  }

  switch (desc.max_group_sz) {
    case 1024:
      DISPATCH_GSZ(1024);
      break;
    case 512:
      DISPATCH_GSZ(512);
      break;
    default:
      DISPATCH_GSZ(256);
      break;
  }

#undef DISPATCH_GSZ
}

} // namespace AtenIpexTypeXPU
} // namespace at
