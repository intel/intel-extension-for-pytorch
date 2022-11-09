#pragma once

#include <utils/DPCPP.h>
#include "SortingCommon.h"
#include "comm/General.h"
#include "comm/KeyTraits.h"

namespace at {
namespace AtenIpexTypeXPU {

#define RADIX_NUMERIC_MIN(A, B) (((A) > (B)) ? (B) : (A))

template <typename T, int STEPS>
inline void subgroup_scan(
    const sycl::sub_group& sg,
    const int sgid,
    const T input,
    T* inclusive_sum,
    T* exclusive_sum) {
  *inclusive_sum = input;
#pragma unroll
  for (int i = 0; i < STEPS; ++i) {
    uint32_t offset = 1u << i;
    T temp = sycl::shift_group_right(sg, *inclusive_sum, offset);
    if (sgid >= offset)
      (*inclusive_sum) += temp;
  }
  *exclusive_sum = (*inclusive_sum) - input;
}

template <typename Type, int COUNTER_LANES, int GROUP_ITEMS, int SUBGROUP_SIZE>
inline Type group_exclusive_sum(
    Type (&storage)[COUNTER_LANES * GROUP_ITEMS],
    sycl::nd_item<1>& item) {
  static_assert(
      GROUP_ITEMS % SUBGROUP_SIZE == 0,
      "GROUP_ITEMS should be n * SUBGROUP_SIZE. (n = 1, 2, 3, ...)");

  const int NUM_SUBGROUPS = GROUP_ITEMS / SUBGROUP_SIZE;
  const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;

  int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();

  int subgroup_local_id = sg.get_local_id();
  int subgroup_id = sg.get_group_linear_id();
  int lane_temp_values[COUNTER_LANES];

  // Read input lane sum
  Type lane_all_sum = 0;

#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    lane_temp_values[lane] = lane_all_sum;
    lane_all_sum += storage[lid * COUNTER_LANES + lane];
  }

  // Get subgroup level exclusive sum
  Type subgroup_inclusive_sum, subgroup_exclusive_sum;
  subgroup_scan<Type, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      lane_all_sum,
      &subgroup_inclusive_sum,
      &subgroup_exclusive_sum);
  item.barrier(dpcpp_local_fence);

  // Write to storage
  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    storage[subgroup_id] = subgroup_inclusive_sum;
  item.barrier(dpcpp_local_fence);

  // Get group prefix
  Type group_all_sum = 0, group_exclusive_sum_;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      group_exclusive_sum_ = group_all_sum;
    group_all_sum += storage[i];
  }
  item.barrier(dpcpp_local_fence);

  // Write to storage
  subgroup_exclusive_sum += group_exclusive_sum_;
#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    storage[lid * COUNTER_LANES + lane] =
        subgroup_exclusive_sum + lane_temp_values[lane];
  }
  item.barrier(dpcpp_local_fence);

  return group_all_sum;
}

template <
    typename KeyT,
    int GROUP_ITEMS,
    int SUBGROUP_SIZE,
    int KEYS_PER_ITEM,
    bool IS_DESCENDING = false,
    typename ValueT = NullType,
    typename DigitT = uint16_t, // Covering GROUP_ITEMS * KEYS_PER_ITEM.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS = 4>
class GroupRadixProcesser {
 public:
  static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
  static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
  static_assert(
      ((1l << (sizeof(DigitT) << 3)) - 1) >= (GROUP_ITEMS * KEYS_PER_ITEM),
      " ");
  using KeyTraitsT = typename KeyTraits<KeyT>::Type;

  enum {
    PROCESSING_LENGTH = GROUP_ITEMS * KEYS_PER_ITEM,
    REG_LEN = KEYS_PER_ITEM,
    RADIX_BUCKETS = 1 << RADIX_BITS,
    KEYS_ONLY = std::is_same<ValueT, NullType>::value,
    PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
    COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
    LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
    DIGIT_BITS = sizeof(DigitT) << 3,
    DIGIT_MASK = (1 << DIGIT_BITS) - 1,
  };

 private:
  union RankT {
    CounterT counters[COUNTER_LANES][GROUP_ITEMS];
    CounterT counters_flat[COUNTER_LANES * GROUP_ITEMS];
    DigitT buckets[COUNTER_LANES][GROUP_ITEMS][PACKING_RATIO];
  };

  union LocalStorage {
    RankT rank_storage;
    KeyTraitsT exchange_ukeys[PROCESSING_LENGTH];
    ValueT exchange_values[PROCESSING_LENGTH];
    int valid_items[GROUP_ITEMS];
  };

  sycl::nd_item<1>& item_;
  const int lid_;
  LocalStorage& local_storage_;

 public:
  static int GetSharedLocalMemorySize() {
    return sizeof(LocalStorage);
  }

  inline GroupRadixProcesser(
      sycl::nd_item<1>& item,
      dpcpp_local_acc_t<unsigned char> buffer)
      : item_(item),
        lid_(item.get_local_id(0)),
        local_storage_(
            reinterpret_cast<LocalStorage&>(*(buffer.get_pointer().get()))) {}

  inline void exchange_keys(
      KeyTraitsT (&ukeys)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM]) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      local_storage_.exchange_ukeys[ranks[ITEM]] = ukeys[ITEM];
    }
    item_.barrier(dpcpp_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      int offset = lid_ * KEYS_PER_ITEM + ITEM;
      ukeys[ITEM] = local_storage_.exchange_ukeys[offset];
    }
    item_.barrier(dpcpp_local_fence);
  }

  inline void exchange_values(
      ValueT (&values)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM]) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      local_storage_.exchange_values[ranks[ITEM]] = values[ITEM];
    }
    item_.barrier(dpcpp_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      int offset = lid_ * KEYS_PER_ITEM + ITEM;
      values[ITEM] = local_storage_.exchange_values[offset];
    }
    item_.barrier(dpcpp_local_fence);
  }

  inline void exchange_keys(
      KeyTraitsT (&ukeys)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int lower_offset,
      int upper_offset,
      uint32_t* mask) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      if (ranks[ITEM] >= lower_offset && ranks[ITEM] < upper_offset) {
        local_storage_.exchange_ukeys[ranks[ITEM] - lower_offset] = ukeys[ITEM];
      }
    }
    item_.barrier(dpcpp_local_fence);
    *mask = 0u;
    int new_length = upper_offset - lower_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      int offset = lid_ * KEYS_PER_ITEM + ITEM;
      if (offset < new_length) {
        *mask |= (1u << ITEM);
        ukeys[ITEM] = local_storage_.exchange_ukeys[offset];
      }
    }
    item_.barrier(dpcpp_local_fence);
  }

  inline void exchange_values(
      ValueT (&values)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int lower_offset,
      int upper_offset) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      if (ranks[ITEM] >= lower_offset && ranks[ITEM] < upper_offset) {
        local_storage_.exchange_values[ranks[ITEM] - lower_offset] =
            values[ITEM];
      }
    }
    item_.barrier(dpcpp_local_fence);
    int new_length = upper_offset - lower_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      int offset = lid_ * KEYS_PER_ITEM + ITEM;
      if (offset < new_length) {
        values[ITEM] = local_storage_.exchange_values[offset];
      }
    }
    item_.barrier(dpcpp_local_fence);
  }

  inline DigitT extract_digit(KeyTraitsT key, int begin, int pass) {
    return ((key >> begin) & ((1 << pass) - 1));
  }

  inline void rank_keys(
      KeyTraitsT (&ukeys)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int begin_bit,
      int pass_bits) {
    DigitT* digit_counters[KEYS_PER_ITEM];
    DigitT sub_counters[KEYS_PER_ITEM];

    // reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
      local_storage_.rank_storage.counters[ITEM][lid_] = 0;
    }
    item_.barrier(dpcpp_local_fence);

#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      auto digit = extract_digit(ukeys[ITEM], begin_bit, pass_bits);
      auto sub_counter = digit >> LOG_COUNTER_LANES;
      auto counter_lane = digit & (COUNTER_LANES - 1);
      if (IS_DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }
      sub_counters[ITEM] = sub_counter;
      digit_counters[ITEM] =
          &local_storage_.rank_storage.buckets[counter_lane][lid_][sub_counter];
      ranks[ITEM] = *digit_counters[ITEM];
      *digit_counters[ITEM] = ranks[ITEM] + 1;
    }
    item_.barrier(dpcpp_local_fence);

    CounterT exclusive = group_exclusive_sum<
        CounterT,
        COUNTER_LANES,
        GROUP_ITEMS,
        SUBGROUP_SIZE>(local_storage_.rank_storage.counters_flat, item_);

    CounterT c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      exclusive = exclusive << DIGIT_BITS;
      c += exclusive;
    }

#pragma unroll
    for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
      local_storage_.rank_storage.counters[INDEX][lid_] += c;
    }
    item_.barrier(dpcpp_local_fence);

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      ranks[ITEM] += *digit_counters[ITEM];
    }
    item_.barrier(dpcpp_local_fence);
  }

  inline void find_select_offset(
      int carry,
      int num_to_select,
      int* out_offset_select,
      int* out_offset_active) {
    *out_offset_select = 0;
    *out_offset_active = 0;
#pragma unroll
    for (int DIGIT = 1; DIGIT < RADIX_BUCKETS; ++DIGIT) {
      auto sub_counter = DIGIT >> LOG_COUNTER_LANES;
      auto counter_lane = DIGIT & (COUNTER_LANES - 1);
      auto count = (int)(local_storage_.rank_storage
                             .buckets[counter_lane][0][sub_counter]);
      if (count > num_to_select) {
        *out_offset_active = count;
        break;
      }
      *out_offset_select = count;
    }
    if (*out_offset_active == 0)
      *out_offset_active = carry;
  }

  inline void rank_keys(
      KeyTraitsT (&ukeys)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int begin_bit,
      int pass_bits,
      uint32_t active_mask,
      int num_to_select,
      int* out_offset_select,
      int* out_offset_active) {
    DigitT* digit_counters[KEYS_PER_ITEM];
    DigitT sub_counters[KEYS_PER_ITEM];

    // reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
      local_storage_.rank_storage.counters[ITEM][lid_] = 0;
    }
    item_.barrier(dpcpp_local_fence);

#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      ranks[ITEM] = PROCESSING_LENGTH;
      if (active_mask >> ITEM & 1) {
        auto digit = extract_digit(ukeys[ITEM], begin_bit, pass_bits);
        auto sub_counter = digit >> LOG_COUNTER_LANES;
        auto counter_lane = digit & (COUNTER_LANES - 1);
        if (IS_DESCENDING) {
          sub_counter = PACKING_RATIO - 1 - sub_counter;
          counter_lane = COUNTER_LANES - 1 - counter_lane;
        }
        sub_counters[ITEM] = sub_counter;
        digit_counters[ITEM] = &local_storage_.rank_storage
                                    .buckets[counter_lane][lid_][sub_counter];
        ranks[ITEM] = *digit_counters[ITEM];
        *digit_counters[ITEM] = ranks[ITEM] + 1;
      }
    }
    item_.barrier(dpcpp_local_fence);

    CounterT exclusive = group_exclusive_sum<
        CounterT,
        COUNTER_LANES,
        GROUP_ITEMS,
        SUBGROUP_SIZE>(local_storage_.rank_storage.counters_flat, item_);

    int carry = 0;
#pragma unroll
    for (int STEP = 0; STEP < PACKING_RATIO; ++STEP) {
      DigitT cc = (exclusive >> (STEP * DIGIT_BITS)) & DIGIT_MASK;
      carry += cc;
    }

    CounterT c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      exclusive = exclusive << DIGIT_BITS;
      c += exclusive;
    }

#pragma unroll
    for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
      local_storage_.rank_storage.counters[INDEX][lid_] += c;
    }
    item_.barrier(dpcpp_local_fence);

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      ranks[ITEM] += *digit_counters[ITEM];
    }
    item_.barrier(dpcpp_local_fence);

    find_select_offset(
        carry, num_to_select, out_offset_select, out_offset_active);

    item_.barrier(dpcpp_local_fence);
  }

  inline void sort_group(
      KeyT (&keys)[KEYS_PER_ITEM],
      ValueT (&values)[KEYS_PER_ITEM],
      int begin_bit,
      int end_bit) {
    KeyTraitsT(&ukeys)[KEYS_PER_ITEM] =
        reinterpret_cast<KeyTraitsT(&)[KEYS_PER_ITEM]>(keys);
    while (true) {
      auto pass_bits = RADIX_NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
      int ranks[KEYS_PER_ITEM];
      rank_keys(ukeys, ranks, begin_bit, pass_bits);
      begin_bit += RADIX_BITS;
      exchange_keys(ukeys, ranks);
      if (!KEYS_ONLY)
        exchange_values(values, ranks);
      if (begin_bit >= end_bit)
        break;
    }
  }

  inline void sort_group(
      KeyT (&keys)[KEYS_PER_ITEM],
      int begin_bit,
      int end_bit) {
    KeyTraitsT(&ukeys)[KEYS_PER_ITEM] =
        reinterpret_cast<KeyTraitsT(&)[KEYS_PER_ITEM]>(keys);
    while (true) {
      auto pass_bits = RADIX_NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
      int ranks[KEYS_PER_ITEM];
      rank_keys(ukeys, ranks, begin_bit, pass_bits);
      begin_bit += RADIX_BITS;
      exchange_keys(ukeys, ranks);
      if (begin_bit >= end_bit)
        break;
    }
  }

  inline void store_keys(
      KeyTraitsT* out,
      KeyTraitsT (&ukeys)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int offset_select,
      int num_selected) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      if (ranks[ITEM] < offset_select) {
        auto ukey = KeyTraits<KeyT>::deconvert(ukeys[ITEM]);
        out[num_selected + ranks[ITEM]] = *reinterpret_cast<KeyTraitsT*>(&ukey);
      }
    }
  }

  inline void store_values(
      ValueT* out,
      ValueT (&values)[KEYS_PER_ITEM],
      int (&ranks)[KEYS_PER_ITEM],
      int offset_select,
      int num_selected) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      if (ranks[ITEM] < offset_select) {
        out[num_selected + ranks[ITEM]] = values[ITEM];
      }
    }
  }

  inline void select_group(
      KeyT (&keys)[KEYS_PER_ITEM],
      ValueT (&values)[KEYS_PER_ITEM],
      int begin_bit,
      int end_bit,
      int num_topk,
      KeyT* out_keys,
      ValueT* out_values) {
    KeyTraitsT(&ukeys)[KEYS_PER_ITEM] =
        reinterpret_cast<KeyTraitsT(&)[KEYS_PER_ITEM]>(keys);
    KeyTraitsT* out_ukeys = reinterpret_cast<KeyTraitsT*>(out_keys);
    uint32_t active_mask = 0xffffffff;
    int num_selected = 0;
    while (true) {
      auto pass_bits = RADIX_NUMERIC_MIN(RADIX_BITS, begin_bit - end_bit);
      begin_bit -= pass_bits;
      int ranks[KEYS_PER_ITEM];
      int offset_select, offset_active;
      rank_keys(
          ukeys,
          ranks,
          begin_bit,
          pass_bits,
          active_mask,
          num_topk - num_selected,
          &offset_select,
          &offset_active);
      if (begin_bit == end_bit)
        offset_select = num_topk - num_selected;
      if (offset_select > 0) {
        store_keys(out_ukeys, ukeys, ranks, offset_select, num_selected);
        if (!KEYS_ONLY)
          store_values(out_values, values, ranks, offset_select, num_selected);
      }
      num_selected += offset_select;
      if (num_selected == num_topk)
        break;
      exchange_keys(ukeys, ranks, offset_select, offset_active, &active_mask);
      if (!KEYS_ONLY)
        exchange_values(values, ranks, offset_select, offset_active);
    }
  }

  inline void select_group(
      KeyT (&keys)[KEYS_PER_ITEM],
      ValueT (&values)[KEYS_PER_ITEM],
      int begin_bit,
      int end_bit,
      int num_topk,
      KeyT threshold,
      KeyT* out_keys,
      ValueT* out_values,
      int* out_num_valids) {
    int num_local_valids = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      if (keys[ITEM] >= threshold)
        num_local_valids++;
    }

    local_storage_.valid_items[lid_] = num_local_valids;
    item_.barrier(dpcpp_local_fence);

    int num_group_valids =
        group_exclusive_sum<int, 1, GROUP_ITEMS, SUBGROUP_SIZE>(
            local_storage_.valid_items, item_);

    int offset = local_storage_.valid_items[lid_];
    item_.barrier(dpcpp_local_fence);

    if (num_group_valids == 0) {
      *out_num_valids = 0;
    } else if (num_group_valids <= num_topk) {
#pragma unroll
      for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
        if (keys[ITEM] >= threshold) {
          out_keys[offset] = keys[ITEM];
          out_values[offset] = values[ITEM];
          offset++;
        }
      }
      *out_num_valids = num_group_valids;
    } else {
      *out_num_valids = num_topk;
      select_group(
          keys, values, begin_bit, end_bit, num_topk, out_keys, out_values);
    }
  }
};

#undef RADIX_NUMERIC_MIN

} // namespace AtenIpexTypeXPU
} // namespace at
