#pragma once

#include <ATen/record_function.h>
#include <utils/DPCPP.h>
#include "Atomics.h"
#include "SimpleReduce.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
DPCPP_DEVICE static inline TensorOptions map_options() {
  if (std::is_same<T, uint8_t>::value)
    return at::TensorOptions().dtype(kByte).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int8_t>::value)
    return at::TensorOptions().dtype(kChar).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int16_t>::value)
    return at::TensorOptions().dtype(kShort).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int32_t>::value)
    return at::TensorOptions().dtype(kInt).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int64_t>::value)
    return at::TensorOptions().dtype(kLong).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, float>::value)
    return at::TensorOptions().dtype(kFloat).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, double>::value)
    return at::TensorOptions().dtype(kDouble).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, at::Half>::value)
    return at::TensorOptions().dtype(kHalf).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, at::BFloat16>::value)
    return at::TensorOptions().dtype(kBFloat16).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, bool>::value)
    return at::TensorOptions().dtype(kBool).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else {
    AT_ERROR("PSTLFunctions: data type cannot be mapped to tensor's dtype.");
  }
  return at::TensorOptions();
}

template <int scan_type, class InputIt, class OutputIt, class T>
DPCPP_DEVICE static inline OutputIt _scan_kernel(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  auto options = map_options<T>();

  if (N <= wgroup_size) {
    // Kogge-Stone addr algorithm;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      dpcpp_local_acc_t<T> local_scan(N, __cgh);

      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();

        // initialize local_input
        auto cur_init = init;
        if (scan_type == 1) {
          local_scan[local_id] = first[local_id];
        } else {
          if (local_id > 0)
            local_scan[local_id] = first[local_id - 1];
          else
            local_scan[local_id] = 0;
        }
        if (local_id == 0)
          local_scan[local_id] += cur_init;
        group_barrier(item_id.get_group());

        // body of KS algo
        for (auto __k = 1; __k < N; __k <<= 1) {
          auto tmp = (local_id >= __k) ? local_scan[local_id - __k] : 0;
          group_barrier(item_id.get_group());
          local_scan[local_id] += tmp;
          group_barrier(item_id.get_group());
        }

        // flush result into dst
        d_first[local_id] = local_scan[local_id];
      };
      __cgh.parallel_for(DPCPP::nd_range</*dim=*/1>(N, N), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return d_first + N;
  }

  Tensor carry = at::empty({ngroups}, options);
  T* carry_ptr = carry.data_ptr<T>();

  // 1. do exclusive_scan on each workgroups
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<T> local_scan(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto global_id = item_id.get_global_linear_id();
      auto group_id = item_id.get_group_linear_id();

      // initialize local_input
      auto cur_init = (group_id == 0 ? init : 0);
      if (global_id < N) {
        if (scan_type == 1) {
          local_scan[local_id] = first[global_id];
        } else {
          if (local_id > 0)
            local_scan[local_id] = first[global_id - 1];
          else
            local_scan[local_id] = 0;
        }
        if (local_id == 0)
          local_scan[local_id] += cur_init;
        if (local_id == wgroup_size - 1) {
          carry_ptr[group_id] = first[global_id];
        }
      }
      group_barrier(item_id.get_group());

      // body of KS algo
      for (auto __k = 1; __k < wgroup_size; __k <<= 1) {
        auto tmp = (local_id >= __k) ? local_scan[local_id - __k] : 0;
        group_barrier(item_id.get_group());
        local_scan[local_id] += tmp;
        group_barrier(item_id.get_group());
      }

      // flush result into dst
      if (global_id < N) {
        d_first[global_id] = local_scan[local_id];
      }
      if (local_id == wgroup_size - 1) {
        if (scan_type == 1)
          carry_ptr[group_id] = local_scan[local_id];
        else
          carry_ptr[group_id] += local_scan[local_id];
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. recursion for carry
  _scan_kernel<0>(carry_ptr, carry_ptr + ngroups, carry_ptr, (T)0);

  // 3. reduce among all work groups and flush data to dst
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<T> local_carry(1, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto global_id = item_id.get_global_linear_id();
      auto group_id = item_id.get_group_linear_id();

      if (local_id == 0)
        local_carry[0] = carry_ptr[group_id];
      group_barrier(item_id.get_group());

      if (global_id < N) {
        d_first[global_id] += local_carry[0];
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  return d_first + N;
}

template <typename T, class InputIt, class OutputIt>
DPCPP_DEVICE static inline OutputIt exclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("exclusive_scan_xpu", {});
  return _scan_kernel<0>(first, last, d_first, init);
}

template <typename T, class InputIt, class OutputIt>
DPCPP_DEVICE static inline OutputIt inclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("inclusive_scan_xpu", {});
  return _scan_kernel<1>(first, last, d_first, init);
}

template <typename index_t, class InputIt, class OutputIt, class UnaryPredicate>
DPCPP_DEVICE static inline OutputIt copy_if(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred) {
  RECORD_FUNCTION("copy_if_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      gmask_ptr[item_id] =
          static_cast<index_t>(static_cast<bool>(pred(first[item_id])));
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (gmask_ptr[item_id] != 0)
        d_first[tpos_ptr[item_id]] = first[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();
  return d_first + M;
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class UnaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt first1,
    InputIt last1,
    OutputIt d_first,
    UnaryOperation unary_op) {
  RECORD_FUNCTION("transform_1_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      d_first[item_id] = static_cast<output_t>(unary_op(first1[item_id]));
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  RECORD_FUNCTION("transform_2_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      d_first[item_id] =
          static_cast<output_t>(binary_op(first1[item_id], first2[item_id]));
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <class T, class ForwardIt>
DPCPP_DEVICE static inline void iota(ForwardIt first, ForwardIt last, T value) {
  RECORD_FUNCTION("iota_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      first[item_id] = value + static_cast<T>(item_id);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T, typename index_t, class ForwardIt, class BinaryPredicate>
ForwardIt unique(ForwardIt first, ForwardIt last, BinaryPredicate p) {
  RECORD_FUNCTION("unique_kernel_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<T>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();

  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (gmask_ptr[item_id] != 0)
        scratchpad_ptr[tpos_ptr[item_id]] = first[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      first[item_id] = scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/>(M), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return first + M;
}

template <
    typename T,
    typename zT,
    typename index_t,
    class ForwardIt,
    class ZipForwardIt,
    class BinaryPredicate>
std::tuple<ForwardIt, ZipForwardIt> unique_with_zip(
    ForwardIt first,
    ForwardIt last,
    ZipForwardIt z_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("unique_with_zip_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<T>();
  auto z_options = map_options<zT>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  Tensor z_scratchpad = at::empty({N}, z_options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();
  zT* z_scratchpad_ptr = z_scratchpad.data_ptr<zT>();

  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (gmask_ptr[item_id] != 0) {
        scratchpad_ptr[tpos_ptr[item_id]] = first[item_id];
        z_scratchpad_ptr[tpos_ptr[item_id]] = z_first[item_id];
      }
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      first[item_id] = scratchpad_ptr[item_id];
      z_first[item_id] = z_scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/>(M), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return std::make_tuple<ForwardIt, ZipForwardIt>(first + M, z_first + M);
}

template <typename output_t, class InputIt, class OutputIt>
OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first) {
  RECORD_FUNCTION("adjacent_difference_1_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<output_t>();
  Tensor scratchpad = at::empty({N}, options);
  output_t* scratchpad_ptr = scratchpad.data_ptr<output_t>();

  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (item_id > 0)
        scratchpad_ptr[item_id] =
            static_cast<output_t>(first[item_id] - first[item_id - 1]);
      else
        scratchpad_ptr[item_id] = static_cast<output_t>(first[item_id]);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  auto cgf_2 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      d_first[item_id] = scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_2);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class BinaryOperation>
OutputIt adjacent_difference(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryOperation op) {
  RECORD_FUNCTION("adjacent_difference_2_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<output_t>();
  Tensor scratchpad = at::empty({N}, options);
  output_t* scratchpad_ptr = scratchpad.data_ptr<output_t>();

  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (item_id > 0)
        scratchpad_ptr[item_id] =
            static_cast<output_t>(op(first[item_id - 1], first[item_id]));
      else
        scratchpad_ptr[item_id] = static_cast<output_t>(first[item_id]);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  auto cgf_2 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      d_first[item_id] = scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_2);

  return d_first + N;
}

template <
    typename input_t,
    typename output_t,
    typename index_t,
    class InputIt,
    class OutputIt,
    class BinaryPredicate>
OutputIt count_by_segment(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("count_by_segment_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<output_t>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with inclusive_scan
  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. calculate counts for each unique point
  Tensor range = at::empty({N + 1}, options);
  output_t* range_ptr = range.data_ptr<output_t>();
  auto range_begin = range_ptr;
  iota(range_begin, range_begin + N + 1, (output_t)0);
  Tensor picked_range = at::empty({N + 1}, options);
  output_t* picked_range_ptr = picked_range.data_ptr<output_t>();
  auto picked_range_begin = picked_range_ptr;
  auto picked_range_end = picked_range_begin;
  picked_range_end = copy_if<index_t>(
      range_begin, range_begin + N, picked_range_begin, [=](output_t a) {
        return gmask_ptr[a] != 0;
      });
  auto num_out = std::distance(picked_range_begin, picked_range_end);
  picked_range[num_out] = N;
  // notice: the temp tensor `range` will be re-used to store the result of
  // adjacent_difference
  adjacent_difference<index_t>(
      picked_range_begin + 1, picked_range_begin + num_out + 1, range_begin);

  // 4. flush range to every elements of counts
  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      d_first[item_id] = range_ptr[tpos_ptr[item_id]];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return d_first + N;
}

} // namespace AtenIpexTypeXPU
} // namespace at
