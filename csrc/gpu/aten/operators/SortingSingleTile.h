#pragma once

#include <ATen/record_function.h>
#include "SortingRadixProcesser.h"
#include "comm/TensorOptions.h"

#ifdef _WIN32
#include <winsock.h>
#undef max
#undef min
#endif

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename KeyType,
    typename ValueType,
    int32_t SUBGROUP_SIZE,
    int32_t GROUP_THREADS,
    int32_t WORKS_PER_ITEM,
    bool IS_DESCENDING = false,
    int32_t RADIX_BITS = 4>
class RadixSortUpsweep {
 public:
  using KeyTraitsT = typename KeyTraits<KeyType>::Type;
  using DigitCounter = u_char;
  using PackedCounter = uint32_t;
  int32_t wi_id;
  int32_t wg_id;
  enum {
    TILE_ITEMS = GROUP_THREADS * WORKS_PER_ITEM,
    BYTES_PER_COUNTER = sizeof(DigitCounter),
    LOG_BYTES_PER_COUNTER = Log2<BYTES_PER_COUNTER>::VALUE,
    PACKING_RATIO = sizeof(PackedCounter) / sizeof(DigitCounter), // 4
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE, // 2
    LOG_COUNTER_LANES = std::max(0, RADIX_BITS - LOG_PACKING_RATIO), // 2
    COUNTER_LANES = 1 << LOG_COUNTER_LANES, // 4

    WARP_THREADS = SUBGROUP_SIZE,
    WARPS = (GROUP_THREADS + WARP_THREADS - 1) / WARP_THREADS,
    LANES_PER_WARP = std::max(1, (COUNTER_LANES + WARPS - 1) / WARPS),

    RADIX_DIGITS = 1 << RADIX_BITS,
  };

 private:
  const KeyType* keys_in;
  int32_t* count;
  const int32_t current_bit;
  const int32_t num_bits;
  sycl::nd_item<1>& item;

  union LocalStorage {
    DigitCounter thread_counters[COUNTER_LANES][GROUP_THREADS]
                                [PACKING_RATIO]; // [4][512][4]
    PackedCounter packed_thread_counters[COUNTER_LANES]
                                        [GROUP_THREADS]; // [4][512]
    int32_t block_counters[WARP_THREADS][RADIX_DIGITS]; // [32][16]
  };

  int32_t local_counts[LANES_PER_WARP][PACKING_RATIO]; // [1][4]
  LocalStorage& local_storage;

 public:
  inline RadixSortUpsweep(
      const KeyType* keys_in,
      int32_t* count,
      const int32_t current_bit,
      const int32_t num_bits,
      sycl::nd_item<1>& item,
      dpcpp_local_acc_t<unsigned char> slm)
      : keys_in(keys_in),
        count(count),
        current_bit(current_bit),
        num_bits(num_bits),
        item(item),
        local_storage(
            reinterpret_cast<LocalStorage&>(*(IPEXGetLocalAccPointer(slm)))) {
    wi_id = item.get_local_id(0);
    wg_id = item.get_group(0);
  }

  static inline int32_t GetSharedLocalStorageSize() {
    return COUNTER_LANES * GROUP_THREADS * sizeof(PackedCounter);
  }

  inline void ProcessFullTile(int32_t wg_offset) {
    KeyTraitsT keys[WORKS_PER_ITEM];
    auto block_ptr = keys_in + wg_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
      keys[ITEM] =
          KeyTraits<KeyType>::convert(block_ptr[wi_id + ITEM * GROUP_THREADS]);
    }
    item.barrier(dpcpp_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
      DigitCounter digit =
          ((keys[ITEM] >> current_bit) & ((1 << num_bits) - 1));
      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage.thread_counters[row_offset][wi_id][sub_counter]++;
    }
  }

  inline void ProcessPartialTile(int32_t wg_offset, int32_t wg_end) {
    // Process partial tile if necessary using single loads
    wg_offset += wi_id;
    while (wg_offset < wg_end) {
      // Load and bucket key
      KeyTraitsT key = KeyTraits<KeyType>::convert(keys_in[wg_offset]);
      DigitCounter digit =
          ((key >> current_bit) & ((1 << num_bits) - 1)); // ExtractDigit

      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage.thread_counters[row_offset][wi_id][sub_counter]++;
      wg_offset += GROUP_THREADS;
    }
  }

  inline void ExtractCounts() {
    int32_t wg_number = item.get_group_range(0);
    int32_t warp_id = wi_id / SUBGROUP_SIZE; // 32
    int32_t warp_tid = wi_id % SUBGROUP_SIZE;
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
      int32_t counter_lane = (LANE * WARPS) + warp_id;
      // Place unpacked digit counters in shared memory
      if (counter_lane < COUNTER_LANES) {
        int32_t digit_row = counter_lane << LOG_PACKING_RATIO;

#pragma unroll
        for (int32_t UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
             UNPACKED_COUNTER++) {
          int32_t bin_idx = digit_row + UNPACKED_COUNTER;
          local_storage.block_counters[warp_tid][bin_idx] =
              local_counts[LANE][UNPACKED_COUNTER];
        }
      }
    }

    item.barrier(dpcpp_local_fence);

    if ((RADIX_DIGITS % GROUP_THREADS != 0) && (wi_id < RADIX_DIGITS)) {
      int32_t bin_idx = wi_id;

      int32_t bin_count = 0;
#pragma unroll
      for (int32_t i = 0; i < WARP_THREADS; ++i)
        bin_count += local_storage.block_counters[i][bin_idx];

      if (IS_DESCENDING)
        bin_idx = RADIX_DIGITS - bin_idx - 1;
      count[(wg_number * bin_idx) + wg_id] = bin_count;
    }
  }

  inline void ResetDigitCounters() {
#pragma unroll
    for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
      local_storage.packed_thread_counters[LANE][wi_id] =
          0; // change to stride=1
  }

  inline void ResetUnpackedCounters() {
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
#pragma unroll
      for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
           UNPACKED_COUNTER++) {
        local_counts[LANE][UNPACKED_COUNTER] = 0;
      }
    }
  }

  inline void UnpackDigitCounts() {
    int32_t warp_id = wi_id / SUBGROUP_SIZE; // 32
    int32_t warp_tid = wi_id % SUBGROUP_SIZE;
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++) {
      const int32_t counter_lane = (LANE * WARPS) + warp_id;
      if (counter_lane < COUNTER_LANES) {
#pragma unroll
        for (int32_t PACKED_COUNTER = 0; PACKED_COUNTER < GROUP_THREADS;
             PACKED_COUNTER += SUBGROUP_SIZE) {
#pragma unroll
          for (int32_t UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
               UNPACKED_COUNTER++) {
            int32_t counter =
                local_storage
                    .thread_counters[counter_lane][warp_tid + PACKED_COUNTER]
                                    [UNPACKED_COUNTER];
            local_counts[LANE][UNPACKED_COUNTER] += counter;
          }
        }
      }
    }
  }
}; // namespace AtenIpexTypeXPU

template <
    typename KeyType,
    typename ValueType,
    int32_t SUBGROUP_SIZE,
    int32_t GROUP_THREADS,
    int32_t WORKS_PER_ITEM,
    bool IS_DESCENDING>
void radix_sort_upsweep_process(
    const KeyType* keys_in,
    int32_t* count, // length = (max_grid_size *
                    // pass_config.radix_digits) +
                    // pass_config.scan_config.tile_size
                    //=(max_wg_number * 16 + 1024 * 4)
    const int sort_sz,
    const int32_t current_bit, // 0, 4, 8, 12
    const int32_t num_bits) {
  using RadixSortUpsweep_t = RadixSortUpsweep<
      KeyType,
      ValueType,
      SUBGROUP_SIZE,
      GROUP_THREADS,
      WORKS_PER_ITEM,
      IS_DESCENDING>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int32_t wg_size = GROUP_THREADS;
  int32_t tile_items = RadixSortUpsweep_t::TILE_ITEMS;
  const auto target_global_size = dpcppMaxWorkItemsPerTile(dev_id);
  const int max_work_group_num = target_global_size / wg_size;
  const int total_tiles = (sort_sz + tile_items - 1) / tile_items;
  const int wg_number = std::min(total_tiles, max_work_group_num);
  int32_t avg_tiles_per_wg = total_tiles / wg_number;
  int32_t big_shares = total_tiles - (avg_tiles_per_wg * wg_number);
  int32_t normal_share_items = avg_tiles_per_wg * tile_items;
  int32_t normal_base_offset = big_shares * tile_items;
  int32_t big_share_items = normal_share_items + tile_items;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto slm = dpcpp_local_acc_t<unsigned char>(
        RadixSortUpsweep_t::GetSharedLocalStorageSize(), cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto Upsweep =
          RadixSortUpsweep_t(keys_in, count, current_bit, num_bits, item, slm);
      int32_t wg_offset, wg_end;
      if (Upsweep.wg_id < big_shares) {
        wg_offset =
            (Upsweep.wg_id *
             big_share_items); // for first several wg, they do one more tile.
        wg_end = wg_offset + big_share_items;
      } else if (Upsweep.wg_id < total_tiles) {
        wg_offset = normal_base_offset + (Upsweep.wg_id * normal_share_items);
        wg_end = std::min(sort_sz, wg_offset + normal_share_items);
      }

      // ResetDigitCounters
      Upsweep.ResetDigitCounters();
      // ResetUnpackedCounters
      Upsweep.ResetUnpackedCounters();

      // Unroll batches of full tiles
      int UNROLL_COUNT = 255 / 4; // the largest value for counter
      int UNROLLED_ELEMENTS = UNROLL_COUNT * tile_items;

      while (wg_offset + UNROLLED_ELEMENTS <= wg_end) {
        for (int i = 0; i < UNROLL_COUNT; ++i) {
          Upsweep.ProcessFullTile(wg_offset);
          wg_offset += tile_items;
        }

        item.barrier(dpcpp_local_fence);

        // Aggregate back into local_count registers to prevent overflow
        Upsweep.UnpackDigitCounts();

        item.barrier(dpcpp_local_fence);
        // clear for next round
        Upsweep.ResetDigitCounters();
      }

      while (wg_offset + tile_items <= wg_end) {
        Upsweep.ProcessFullTile(wg_offset);
        wg_offset += tile_items;
      }

      Upsweep.ProcessPartialTile(wg_offset, wg_end);
      item.barrier(dpcpp_local_fence);
      Upsweep.UnpackDigitCounts();
      item.barrier(dpcpp_local_fence);
      Upsweep.ExtractCounts();
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(wg_number * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int32_t SUBGROUP_SIZE, int32_t WORKS_PER_ITEM>
inline void ConsumePartialTile(
    int32_t* count,
    sycl::nd_item<1>& item,
    int32_t wg_offset,
    int32_t tile_num,
    int32_t running_prefix,
    int32_t* slm) {
  // running_prefix for next round
  // 1. load
  int32_t partial_output[WORKS_PER_ITEM];
  auto wi_id = item.get_local_id(0);
  auto d_local = count + wg_offset;
#pragma unroll
  for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
    if ((wi_id * WORKS_PER_ITEM) + ITEM < tile_num) {
      partial_output[ITEM] = d_local[(wi_id * WORKS_PER_ITEM) + ITEM];
    } else {
      partial_output[ITEM] = *d_local;
    }
  }

  item.barrier(dpcpp_local_fence);

  // 2.1 thread reduce
  int32_t thread_partial = partial_output[0];
#pragma unroll
  for (int i = 1; i < WORKS_PER_ITEM; ++i) {
    thread_partial = thread_partial + partial_output[i];
  }

  // 2.2 scan
  int32_t subgroup_inclusive_sum, subgroup_exclusive_sum;
  auto sg = item.get_sub_group();
  const int32_t subgroup_local_id = wi_id % SUBGROUP_SIZE;
  const int32_t subgroup_id = wi_id / SUBGROUP_SIZE;
  const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;
  subgroup_scan<int32_t, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      thread_partial,
      &subgroup_inclusive_sum,
      &subgroup_exclusive_sum);

  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    slm[subgroup_id] = subgroup_inclusive_sum;
  item.barrier(dpcpp_local_fence);

  int32_t block_all_sum = 0, warp_prefix_sum;
  const int32_t NUM_SUBGROUPS = item.get_local_range(0) / SUBGROUP_SIZE;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      warp_prefix_sum = block_all_sum;
    block_all_sum += slm[i]; // 0
  }

  subgroup_exclusive_sum += warp_prefix_sum;
  // running_prefix is a global value to record before work group sum
  // only work item 0 compute prefix
  subgroup_exclusive_sum += running_prefix;

  if (wi_id == 0)
    running_prefix += block_all_sum;
  // finish exclusion scan

  // 2.3 reduce value split into each item by one thread
  int32_t inclusive = partial_output[0];
  inclusive = subgroup_exclusive_sum + inclusive;
  partial_output[0] = subgroup_exclusive_sum;
  int32_t exclusive = inclusive;
#pragma unroll
  for (int i = 1; i < WORKS_PER_ITEM; ++i) {
    inclusive = exclusive + partial_output[i];
    partial_output[i] = exclusive;
    exclusive = inclusive;
  }

  // 3 store, register back to global memory
#pragma unroll
  for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
    if (wi_id * WORKS_PER_ITEM + ITEM < tile_num) {
      d_local[(wi_id * WORKS_PER_ITEM) + ITEM] = partial_output[ITEM];
    }
  }
}

template <int32_t SUBGROUP_SIZE, int32_t WORKS_PER_ITEM>
inline void ConsumeTile(
    int32_t* count,
    sycl::nd_item<1>& item,
    int32_t wg_offset,
    int32_t running_prefix,
    int32_t* slm) {
  // running_prefix for next round
  // 1. load
  int32_t partial_output[WORKS_PER_ITEM];
  auto wi_id = item.get_local_id(0);
  auto d_local = count + wg_offset;
#pragma unroll
  for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
    partial_output[ITEM] = d_local[(wi_id * WORKS_PER_ITEM) + ITEM];
  }
  item.barrier(dpcpp_local_fence);

  // 2.1 thread reduce
  int32_t thread_partial = partial_output[0];
#pragma unroll
  for (int i = 1; i < WORKS_PER_ITEM; ++i) {
    thread_partial = thread_partial + partial_output[i];
  }

  // 2.2 scan
  int32_t subgroup_inclusive_sum, subgroup_exclusive_sum;
  auto sg = item.get_sub_group();
  const int32_t subgroup_local_id = wi_id % SUBGROUP_SIZE;
  const int32_t subgroup_id = wi_id / SUBGROUP_SIZE;
  const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;
  subgroup_scan<int32_t, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      thread_partial,
      &subgroup_inclusive_sum,
      &subgroup_exclusive_sum);

  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    slm[subgroup_id] = subgroup_inclusive_sum;
  item.barrier(dpcpp_local_fence);

  int32_t block_all_sum = 0, warp_prefix_sum;
  const int32_t NUM_SUBGROUPS = item.get_local_range(0) / SUBGROUP_SIZE;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      warp_prefix_sum = block_all_sum;
    block_all_sum += slm[i]; // 0
  }

  subgroup_exclusive_sum += warp_prefix_sum;
  // running_prefix is a global value to record before work group sum
  // only work item 0 compute prefix
  subgroup_exclusive_sum += running_prefix;

  if (wi_id == 0)
    running_prefix += block_all_sum;
  // finish exclusion scan

  // 2.3 reduce value split into each item by one thread
  int32_t inclusive = partial_output[0];
  inclusive = subgroup_exclusive_sum + inclusive;
  partial_output[0] = subgroup_exclusive_sum;
  int32_t exclusive = inclusive;
#pragma unroll
  for (int i = 1; i < WORKS_PER_ITEM; ++i) {
    inclusive = exclusive + partial_output[i];
    partial_output[i] = exclusive;
    exclusive = inclusive;
  }

  // 3 store, register back to global memory
#pragma unroll
  for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
    d_local[(wi_id * WORKS_PER_ITEM) + ITEM] = partial_output[ITEM];
  }
}

template <int32_t SUBGROUP_SIZE, int32_t WORKS_PER_ITEM>
void RadixSortScanBins(
    int32_t* count, // length = (max_grid_size *
                    // pass_config.radix_digits) +
                    // pass_config.scan_config.tile_size
                    //=(max_wg_number * 16 + 1024 * 4)
    int num_counts) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t wg_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto slm = dpcpp_local_acc_t<int32_t>(wg_size / SUBGROUP_SIZE, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      int wg_offset = 0;
      int32_t running_prefix = 0;
      const int32_t TILE_ITEMS = WORKS_PER_ITEM * wg_size;
      while (wg_offset + TILE_ITEMS <= num_counts) {
        ConsumeTile<SUBGROUP_SIZE, WORKS_PER_ITEM>(
            count,
            item,
            wg_offset,
            running_prefix,
            (int32_t*)IPEXGetLocalAccPointer(slm));
        wg_offset += TILE_ITEMS;
      }

      if (wg_offset < num_counts) {
        ConsumePartialTile<SUBGROUP_SIZE, WORKS_PER_ITEM>(
            count,
            item,
            wg_offset,
            num_counts - wg_offset,
            running_prefix,
            (int32_t*)IPEXGetLocalAccPointer(slm));
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename KeyType,
    typename ValueType,
    int32_t SUBGROUP_SIZE,
    int32_t GROUP_THREADS,
    int32_t WORKS_PER_ITEM,
    bool IS_DESCENDING = false,
    int RADIX_BITS = 4>
class RadixSortDownsweep {
 public:
  using KeyTraitsT = typename KeyTraits<KeyType>::Type;
  using DigitCounter = uint16_t;
  using PackedCounter = uint32_t;
  int wi_id;
  int wg_id;
  enum {
    TILE_ITEMS = WORKS_PER_ITEM * GROUP_THREADS,
    BYTES_PER_COUNTER = sizeof(DigitCounter),
    LOG_BYTES_PER_COUNTER = Log2<BYTES_PER_COUNTER>::VALUE,
    PACKING_RATIO = sizeof(PackedCounter) / sizeof(DigitCounter), // 4
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE, // 2
    LOG_COUNTER_LANES = std::max(0, RADIX_BITS - LOG_PACKING_RATIO), // 2
    COUNTER_LANES = 1 << LOG_COUNTER_LANES, // 4

    WARP_THREADS = SUBGROUP_SIZE,
    WARPS = (GROUP_THREADS + WARP_THREADS - 1) / WARP_THREADS,

    RADIX_DIGITS = 1 << RADIX_BITS,
    KEY_TRAITS_TYPE_MASK = 1l << ((sizeof(KeyTraitsT) << 3) - 1),
  };

 private:
  const KeyType* keys_in;
  KeyType* keys_out;
  const ValueType* values_in;
  ValueType* values_out;
  int32_t* count;
  const int sort_sz;
  const int32_t current_bit;
  const int32_t num_bits;
  sycl::nd_item<1>& item;
  int32_t bin_offset;

  union RankT {
    DigitCounter buckets[COUNTER_LANES][GROUP_THREADS]
                        [PACKING_RATIO]; // [4][512][4]
    PackedCounter scan_storage[COUNTER_LANES][GROUP_THREADS]; // [4][512]
    PackedCounter scan_flat[COUNTER_LANES * GROUP_THREADS];
  };

  union LocalStorage {
    RankT rank_storage;
    struct {
      KeyTraitsT exchange_keys[TILE_ITEMS];
      int32_t relative_bin_offsets[RADIX_DIGITS];
    };
    ValueType exchange_values[TILE_ITEMS];
    int32_t exclusive_digit_prefix[RADIX_DIGITS];
  };

  LocalStorage& local_storage;

 public:
  inline RadixSortDownsweep(
      const KeyType* keys_in,
      KeyType* keys_out,
      const ValueType* values_in,
      ValueType* values_out,
      int32_t* count,
      const int sort_sz,
      const int32_t current_bit,
      const int32_t num_bits,
      sycl::nd_item<1>& item,
      dpcpp_local_acc_t<unsigned char> slm)
      : keys_in(keys_in),
        keys_out(keys_out),
        values_in(values_in),
        values_out(values_out),
        count(count),
        sort_sz(sort_sz),
        current_bit(current_bit),
        num_bits(num_bits),
        item(item),
        local_storage(
            reinterpret_cast<LocalStorage&>(*(IPEXGetLocalAccPointer(slm)))) {
    wi_id = item.get_local_id(0);
    wg_id = item.get_group(0);
    int32_t bin_idx = wi_id;
    if ((GROUP_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS)) {
      if (IS_DESCENDING)
        bin_idx = RADIX_DIGITS - bin_idx - 1;
      bin_offset = count[(item.get_group_range(0) * bin_idx) + wg_id];
    }
    item.barrier(dpcpp_local_fence);
  }

  static inline int32_t GetSharedLocalStorageSize() {
    return sizeof(LocalStorage);
  }

  inline void LoadKeys(
      KeyTraitsT (&keys)[WORKS_PER_ITEM],
      int32_t wg_offset,
      const int32_t valid_items) {
    KeyTraitsT PADDING_KEY;
    if (IS_DESCENDING) {
      PADDING_KEY = 0;
    } else {
      PADDING_KEY = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
      PADDING_KEY = PADDING_KEY ^ (PADDING_KEY - 1);
    }
    auto keys_local = keys_in + wg_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
      int offset = (wi_id * WORKS_PER_ITEM) + ITEM;
      keys[ITEM] = (offset < valid_items)
          ? KeyTraits<KeyType>::convert(keys_local[offset])
          : PADDING_KEY;
    }
    item.barrier(dpcpp_local_fence);
  }

  inline void LoadValues(
      ValueType (&values)[WORKS_PER_ITEM],
      int32_t wg_offset,
      const int32_t valid_items) {
    auto values_local = values_in + wg_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ITEM++) {
      int offset = (wi_id * WORKS_PER_ITEM) + ITEM;
      if (offset < valid_items)
        values[ITEM] = values_local[offset];
    }
    item.barrier(dpcpp_local_fence);
  }

  inline DigitCounter ExtractDigit(KeyTraitsT key, int begin, int pass) {
    return ((key >> begin) & ((1 << pass) - 1));
  }

  inline void RankKeys(
      KeyTraitsT (&key)[WORKS_PER_ITEM],
      int32_t (&rank)[WORKS_PER_ITEM],
      int32_t begin_bit,
      int32_t pass_bits,
      int32_t& exclusive_digit_prefix) {
    DigitCounter* digit_counters[WORKS_PER_ITEM];
    DigitCounter sub_counters[WORKS_PER_ITEM];

    // Reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM)
      local_storage.rank_storage.scan_storage[ITEM][wi_id] = 0;
    item.barrier(dpcpp_local_fence);

    // Bin
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      auto digit = ExtractDigit(key[ITEM], begin_bit, pass_bits);
      auto sub_counter = digit >> LOG_COUNTER_LANES;
      auto counter_lane = digit & (COUNTER_LANES - 1);
      if (IS_DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }
      sub_counters[ITEM] = sub_counter;
      digit_counters[ITEM] =
          &local_storage.rank_storage.buckets[counter_lane][wi_id][sub_counter];
      rank[ITEM] = *digit_counters[ITEM];
      *digit_counters[ITEM] = rank[ITEM] + 1;
    }
    item.barrier(dpcpp_local_fence);

    // Exclusive scan
    PackedCounter temp = group_exclusive_sum<
        PackedCounter,
        COUNTER_LANES,
        GROUP_THREADS,
        SUBGROUP_SIZE>(local_storage.rank_storage.scan_flat, item);

    // Decode packing data
    PackedCounter c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      c += temp << (sizeof(DigitCounter) * 8 * STEP);
    }
    // unpack the counting data
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ITEM++) {
      local_storage.rank_storage.scan_storage[ITEM][wi_id] += c;
    }
    item.barrier(dpcpp_local_fence);

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      rank[ITEM] += *digit_counters[ITEM];
    }
    item.barrier(dpcpp_local_fence);

    int32_t bin_idx = wi_id;
    if ((GROUP_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS)) {
      if (IS_DESCENDING)
        bin_idx = RADIX_DIGITS - bin_idx - 1;

      uint32_t counter_lane = (bin_idx & (COUNTER_LANES - 1));
      uint32_t sub_counter = bin_idx >> (LOG_COUNTER_LANES);
      exclusive_digit_prefix =
          local_storage.rank_storage.buckets[counter_lane][0][sub_counter];
    }
    item.barrier(dpcpp_local_fence);
  }

  template <bool FULL_TILE>
  void ScatterKeys(
      KeyTraitsT (&twiddled_keys)[WORKS_PER_ITEM],
      int32_t (&relative_bin_offsets)[WORKS_PER_ITEM],
      int32_t (&ranks)[WORKS_PER_ITEM],
      int32_t valid_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      local_storage.exchange_keys[ranks[ITEM]] = twiddled_keys[ITEM];
    }
    item.barrier(dpcpp_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      KeyTraitsT key =
          local_storage.exchange_keys[wi_id + (ITEM * GROUP_THREADS)];
      auto digit = ExtractDigit(key, current_bit, num_bits);
      relative_bin_offsets[ITEM] = local_storage.relative_bin_offsets[digit];

      if (FULL_TILE ||
          (static_cast<int32_t>(wi_id + (ITEM * GROUP_THREADS)) <
           valid_items)) {
        keys_out[relative_bin_offsets[ITEM] + wi_id + (ITEM * GROUP_THREADS)] =
            KeyTraits<KeyType>::deconvert(key);
      }
    }
  }

  template <bool FULL_TILE>
  void GatherScatterValues(
      int32_t (&relative_bin_offsets)[WORKS_PER_ITEM],
      int32_t (&ranks)[WORKS_PER_ITEM],
      int32_t wg_offset,
      int32_t valid_items) {
    ValueType values[WORKS_PER_ITEM];
    LoadValues(values, wg_offset, valid_items);

#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      local_storage.exchange_values[ranks[ITEM]] = values[ITEM];
    }
    item.barrier(dpcpp_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < WORKS_PER_ITEM; ++ITEM) {
      ValueType value =
          local_storage.exchange_values[wi_id + (ITEM * GROUP_THREADS)];

      if (FULL_TILE ||
          (static_cast<int32_t>(wi_id + (ITEM * GROUP_THREADS)) <
           valid_items)) {
        values_out
            [relative_bin_offsets[ITEM] + wi_id + (ITEM * GROUP_THREADS)] =
                value;
      }
    }
  }

  template <bool FULL_TILE>
  inline void ProcessTile(
      int32_t wg_offset,
      const int32_t valid_items = TILE_ITEMS) {
    KeyTraitsT keys[WORKS_PER_ITEM];
    int32_t ranks[WORKS_PER_ITEM];
    int32_t relative_bin_offsets[WORKS_PER_ITEM];
    LoadKeys(keys, wg_offset, valid_items);

    int32_t exclusive_digit_prefix;
    RankKeys(keys, ranks, current_bit, num_bits, exclusive_digit_prefix);

    // copy exclusive_digit_prefix to SLM
    if ((GROUP_THREADS == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      local_storage.exclusive_digit_prefix[wi_id] = exclusive_digit_prefix;
    }
    item.barrier(dpcpp_local_fence);

    // Get inclusive digit prefix
    int32_t inclusive_digit_prefix;
    if ((GROUP_THREADS == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      if (IS_DESCENDING) {
        // Get inclusive digit prefix from exclusive prefix (higher bins come
        // first)
        inclusive_digit_prefix = (wi_id == 0)
            ? (GROUP_THREADS * WORKS_PER_ITEM)
            : local_storage.exclusive_digit_prefix[wi_id - 1];
      } else {
        // Get inclusive digit prefix from exclusive prefix (lower bins come
        // first)
        inclusive_digit_prefix = (wi_id == RADIX_DIGITS - 1)
            ? (GROUP_THREADS * WORKS_PER_ITEM)
            : local_storage.exclusive_digit_prefix[wi_id + 1];
      }
    }
    item.barrier(dpcpp_local_fence);

    if ((GROUP_THREADS == RADIX_DIGITS) || (wi_id < RADIX_DIGITS)) {
      bin_offset -= exclusive_digit_prefix;
      local_storage.relative_bin_offsets[wi_id] = bin_offset;
      bin_offset += inclusive_digit_prefix;
    }

    ScatterKeys<FULL_TILE>(
        keys,
        relative_bin_offsets,
        ranks,
        valid_items); // valid_items = 4 x wg_size
    GatherScatterValues<FULL_TILE>(
        relative_bin_offsets, ranks, wg_offset, valid_items);
  }

  inline void ProcessRegion(int32_t wg_offset, int32_t wg_end) {
#pragma unroll
    while (wg_offset + TILE_ITEMS <= wg_end) {
      ProcessTile<true>(wg_offset);
      wg_offset += TILE_ITEMS;

      item.barrier(dpcpp_local_fence);
    }

    if (wg_offset < wg_end) {
      ProcessTile<false>(wg_offset, wg_end - wg_offset);
    }
  }
};

template <
    typename KeyType,
    typename ValueType,
    int32_t SUBGROUP_SIZE,
    int32_t GROUP_THREADS,
    int32_t WORKS_PER_ITEM,
    bool IS_DESCENDING>
void radix_sort_downsweep_process(
    const KeyType* keys_in,
    KeyType* keys_out,
    const ValueType* values_in,
    ValueType* values_out,
    int32_t* count,
    const int sort_sz,
    const int32_t current_bit, // 0, 4, 8, 12
    const int32_t num_bits) {
  using RadixSortDownsweep_t = RadixSortDownsweep<
      KeyType,
      ValueType,
      SUBGROUP_SIZE,
      GROUP_THREADS,
      WORKS_PER_ITEM,
      IS_DESCENDING>;
  using KeyTraitsT = typename RadixSortDownsweep_t::KeyTraitsT;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int32_t wg_size = GROUP_THREADS;
  const auto target_global_size = dpcppMaxWorkItemsPerTile(dev_id);
  const int32_t max_work_group_num = target_global_size / wg_size;
  const int32_t tile_items = RadixSortDownsweep_t::TILE_ITEMS;
  const int32_t total_tiles = (sort_sz + tile_items - 1) / tile_items;
  const int32_t wg_number = std::min(total_tiles, max_work_group_num);
  int32_t avg_tiles_per_wg = total_tiles / wg_number;
  int32_t big_shares = total_tiles - (avg_tiles_per_wg * wg_number);
  int32_t normal_share_items = avg_tiles_per_wg * tile_items;
  int32_t normal_base_offset = big_shares * tile_items;
  int32_t big_share_items = normal_share_items + tile_items;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto slm = dpcpp_local_acc_t<unsigned char>(
        RadixSortDownsweep_t::GetSharedLocalStorageSize(), cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
      auto Downsweep = RadixSortDownsweep_t(
          keys_in,
          keys_out,
          values_in,
          values_out,
          count,
          sort_sz,
          current_bit,
          num_bits,
          item,
          slm);

      int32_t wg_offset, wg_end;
      if (Downsweep.wg_id < big_shares) {
        wg_offset =
            (Downsweep.wg_id * big_share_items); // for first several
                                                 // wg, they do one more tile.
        wg_end = wg_offset + big_share_items;
      } else if (Downsweep.wg_id < total_tiles) {
        wg_offset = normal_base_offset + (Downsweep.wg_id * normal_share_items);
        wg_end = std::min(sort_sz, wg_offset + normal_share_items);
      }
      Downsweep.ProcessRegion(wg_offset, wg_end);
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(wg_number * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename KeyType,
    typename ValueType,
    bool IS_DESCENDING,
    int32_t SUBGROUP_SIZE,
    int32_t GROUP_THREADS>
void radix_sort_iteration_impl(
    const int32_t sort_sz,
    const int32_t current_bit,
    const int32_t num_bits,
    const KeyType* keys_in,
    const ValueType* values_in,
    KeyType* keys_out,
    ValueType* values_out,
    int32_t* count,
    const int32_t count_sz) {
  radix_sort_upsweep_process<
      KeyType,
      ValueType,
      SUBGROUP_SIZE,
      GROUP_THREADS,
      4,
      IS_DESCENDING>(keys_in, count, sort_sz, current_bit, num_bits);
  RadixSortScanBins<SUBGROUP_SIZE, 4>(count, count_sz);
  radix_sort_downsweep_process<
      KeyType,
      ValueType,
      SUBGROUP_SIZE,
      GROUP_THREADS,
      4,
      IS_DESCENDING>(
      keys_in,
      keys_out,
      values_in,
      values_out,
      count,
      sort_sz,
      current_bit,
      num_bits);
}

template <typename KeyType, typename ValueType, bool IS_DESCENDING>
void radix_sort_iteration(
    const int32_t sort_sz,
    const int32_t current_bit,
    const int32_t num_bits,
    const KeyType* keys_in,
    const ValueType* values_in,
    KeyType* keys_out,
    ValueType* values_out,
    int32_t* count,
    const int32_t count_sz) {
  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());

#define DISPATCH_RADIX_SORT_IMPI(SG_SIZE, WG_SIZE) \
  radix_sort_iteration_impl<                       \
      KeyType,                                     \
      ValueType,                                   \
      IS_DESCENDING,                               \
      SG_SIZE,                                     \
      WG_SIZE>(                                    \
      sort_sz,                                     \
      current_bit,                                 \
      num_bits,                                    \
      keys_in,                                     \
      values_in,                                   \
      keys_out,                                    \
      values_out,                                  \
      count,                                       \
      count_sz);

  if (dpcppMaxWorkGroupSize() < 512) {
    switch (dev_prop->subgroup_sizes[0] * 2) {
      case 32:
        DISPATCH_RADIX_SORT_IMPI(32, 256)
        break;
      default:
        DISPATCH_RADIX_SORT_IMPI(16, 256)
    }
  } else {
    switch (dev_prop->subgroup_sizes[0] * 2) {
      case 32:
        DISPATCH_RADIX_SORT_IMPI(32, 512)
        break;
      default:
        DISPATCH_RADIX_SORT_IMPI(16, 512)
    }
  }
}

template <typename KeyType, typename ValueType, bool IS_DESCENDING>
void radix_sort_single_tile(KeyType* key, ValueType* val, const int sort_sz) {
  RECORD_FUNCTION("radix_sort_single_tile", {});
  const int32_t radix_bits = 4;
  const int32_t radix_iters = (sizeof(KeyType) * 8) / radix_bits;

  const int32_t radix_states = 1 << radix_bits;
  const int32_t wg_size = dpcppMaxWorkGroupSize() < 512 ? 256 : 512;
  const auto target_global_size = dpcppMaxWorkItemsPerTile();
  const int max_work_group_num = target_global_size / wg_size;
  const int64_t count_sz = max_work_group_num * radix_states;
  auto count_options = map_options<int32_t>();
  auto key_options = map_options<KeyType>();
  auto val_options = map_options<ValueType>();
  Tensor count_tensor = at::empty({count_sz}, count_options);
  Tensor tmp_key_tensor = at::empty({sort_sz}, key_options);
  Tensor tmp_value_tensor = at::empty({sort_sz}, val_options);
  int32_t* count = count_tensor.data_ptr<int32_t>();
  KeyType* tmp_key = tmp_key_tensor.data_ptr<KeyType>();
  ValueType* tmp_value = tmp_value_tensor.data_ptr<ValueType>();
  int32_t current_bit = 0;
  for (int32_t radix_iter = 0; radix_iter < radix_iters; ++radix_iter) {
    current_bit = radix_iter * radix_bits;
    if (radix_iter % 2 == 0) {
      radix_sort_iteration<KeyType, ValueType, IS_DESCENDING>(
          sort_sz,
          current_bit,
          radix_bits,
          key,
          val,
          tmp_key,
          tmp_value,
          count,
          count_sz);
    } else {
      radix_sort_iteration<KeyType, ValueType, IS_DESCENDING>(
          sort_sz,
          current_bit,
          radix_bits,
          tmp_key,
          tmp_value,
          key,
          val,
          count,
          count_sz);
    }
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at