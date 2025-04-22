#pragma once

#include <sycl/sycl.hpp>
#include <cmath>

#define MAX_SUBGROUP_SIZE 32

using namespace sycl;

namespace sycl_scan {

template <int A>
struct Int2Type {
  static constexpr int value = A;
};

template <typename T>
inline T shuffle_up(sycl::sub_group sg, T value, int delta) {
  int lane_id = sg.get_local_linear_id();
  if (lane_id < delta) {
    delta = 0;
  }
  return sycl::select_from_group(sg, value, lane_id - delta);
}

template <int LENGTH, typename T, typename ReductionOp>
inline T work_item_reduce(T (&input)[LENGTH], ReductionOp reduce_op) {
  T result = input[0];
#pragma unroll
  for (int i = 1; i < LENGTH; ++i) {
    result = reduce_op(result, input[i]);
  }

  return result;
}

template <int LENGTH, typename T, typename ScanOp>
inline T work_item_inclusive_scan(
    T inclusive,
    T* input,
    T* output,
    ScanOp scan_op,
    Int2Type<LENGTH>) {
#pragma unroll
  for (int i = 0; i < LENGTH; ++i) {
    inclusive = scan_op(inclusive, input[i]);
    output[i] = inclusive;
  }
  return inclusive;
}

template <int LENGTH, typename T, typename ScanOp>
inline T work_item_inclusive_scan(
    T* input,
    T* output,
    ScanOp scan_op,
    T prefix,
    bool apply_prefix = true) {
  T inclusive = input[0];
  if (apply_prefix) {
    inclusive = scan_op(prefix, inclusive);
  }
  output[0] = inclusive;

  return work_item_inclusive_scan(
      inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
}

template <int LENGTH, typename T, typename ScanOp>
inline T work_item_inclusive_scan(
    T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    T prefix,
    bool apply_prefix = true) {
  return work_item_inclusive_scan<LENGTH>(
      (T*)input, (T*)output, scan_op, prefix, apply_prefix);
}

// implementation of inclusive_scan_stem by shuffle
template <typename T, typename ScanOp>
inline T inclusive_scan_over_subgroup_step(
    sycl::sub_group sg,
    T input,
    ScanOp scan_op,
    int offset) {
  int lane_id = sg.get_local_linear_id();
  T temp = shuffle_up(sg, input, offset);

  T output = scan_op(temp, input);

  if (lane_id < offset) {
    output = input;
  }

  return output;
}

// sub_group scan
template <typename T, typename ScanOp>
inline void inclusive_scan_over_subgroup(
    sycl::sub_group sg,
    T input,
    T& inclusive_output,
    ScanOp scan_op) {
  auto sub_group_range = sg.get_local_linear_range();

  int STEPS = sycl::log2(static_cast<float>(sub_group_range));

  inclusive_output = input;
#pragma unroll
  for (int STEP = 0; STEP < STEPS; ++STEP) {
    inclusive_output = inclusive_scan_over_subgroup_step(
        sg, inclusive_output, scan_op, (1 << STEP));
  }
}

template <int RANGE_DIM, typename T, typename ScanOp>
inline T get_subgroup_prefix(
    sycl::group<RANGE_DIM> group,
    sycl::sub_group sg,
    T subgroup_aggregate,
    ScanOp scan_op,
    T& group_aggregate) {
  auto lane_id = sg.get_local_linear_id();
  auto subgroup_id = sg.get_group_linear_id();
  auto subgroup_range = sg.get_group_linear_range();
  auto subgroup_local_range = sg.get_local_linear_range();

  // Use shared memory to store the subgroup aggregate
  auto& subgroup_aggregates =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<
          T[MAX_SUBGROUP_SIZE]>(group);

  if (lane_id == subgroup_local_range - 1) {
    subgroup_aggregates[subgroup_id] = subgroup_aggregate;
  }

  group_barrier(group);

  group_aggregate = subgroup_aggregates[0];

  T subgroup_prefix;
#pragma unroll
  for (int subgroup_offset = 1; subgroup_offset < subgroup_range;
       ++subgroup_offset) {
    if (subgroup_id == subgroup_offset) {
      subgroup_prefix = group_aggregate;
    }
    group_aggregate =
        scan_op(group_aggregate, subgroup_aggregates[subgroup_offset]);
  }

  return subgroup_prefix;
}

template <int RANGE_DIM, typename T, typename ScanOp>
inline T inclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T input,
    ScanOp scan_op,
    T& group_aggregate) {
  T inclusive_output;
  auto group = item.get_group();
  auto sg = item.get_sub_group();
  inclusive_scan_over_subgroup(sg, input, inclusive_output, scan_op);

  T subgroup_prefix = get_subgroup_prefix(
      group, sg, inclusive_output, scan_op, group_aggregate);
  auto subgroup_id = sg.get_group_linear_id();

  if (subgroup_id != 0) {
    inclusive_output = scan_op(subgroup_prefix, inclusive_output);
  }

  return inclusive_output;
}

template <int RANGE_DIM, typename T, typename ScanOp>
inline T exclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T input,
    ScanOp scan_op,
    T& group_aggregate) {
  auto sg = item.get_sub_group();

  T inclusive_output;
  inclusive_scan_over_subgroup(sg, input, inclusive_output, scan_op);

  int lane_id = sg.get_local_linear_id();
  T exclusive_output = shuffle_up(sg, inclusive_output, 1);

  auto group = item.get_group();

  T subgroup_prefix = get_subgroup_prefix(
      group, sg, inclusive_output, scan_op, group_aggregate);
  auto subgroup_id = sg.get_group_linear_id();

  if (subgroup_id != 0) {
    exclusive_output = scan_op(subgroup_prefix, exclusive_output);

    if (lane_id == 0) {
      exclusive_output = subgroup_prefix;
    }
  }

  return exclusive_output;
}

template <
    int RANGE_DIM,
    typename T,
    typename ScanOp,
    typename BlockPrefixCallbackOp>
inline void inclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T input,
    T& output,
    ScanOp scan_op,
    BlockPrefixCallbackOp& prefix_callback_op) {
  T group_aggregate;
  output = inclusive_scan_over_group(item, input, scan_op, group_aggregate);

  auto block_prefix = prefix_callback_op(group_aggregate);
  output = scan_op(block_prefix, output);
}

template <
    int RANGE_DIM,
    typename T,
    typename ScanOp,
    typename BlockPrefixCallbackOp>
inline void exclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T input,
    T& output,
    ScanOp scan_op,
    BlockPrefixCallbackOp& prefix_callback_op) {
  T group_aggregate;
  output = exclusive_scan_over_group(item, input, scan_op, group_aggregate);

  auto block_prefix = prefix_callback_op(group_aggregate);
  if (item.get_local_linear_id() == 0) {
    output = block_prefix;
  } else {
    output = scan_op(block_prefix, output);
  }
}

template <
    int RANGE_DIM,
    int ITEMS_PER_THREAD,
    typename T,
    typename ScanOp,
    typename BlockPrefixCallbackOp>
inline void inclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item,
    T (&input)[ITEMS_PER_THREAD],
    T (&output)[ITEMS_PER_THREAD],
    ScanOp scan_op,
    BlockPrefixCallbackOp& prefix_callback_op) {
  if (ITEMS_PER_THREAD == 1) {
    inclusive_scan_over_group(
        item, input[0], output[0], scan_op, prefix_callback_op);
  } else {
    T work_item_prefix = work_item_reduce(input, scan_op);

    exclusive_scan_over_group(
        item, work_item_prefix, work_item_prefix, scan_op, prefix_callback_op);

    auto linear_id = item.get_local_linear_id();
    work_item_inclusive_scan(input, output, scan_op, work_item_prefix);
  }
}
} // namespace sycl_scan
