#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "MemoryAccess.h"
#include "comm/Numerics.h"
#include "comm/TensorOptions.h"

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename T>
inline void swap_var(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename KeyType, typename ValueType, typename CompFunc>
inline void compare_and_swap(
    KeyType& kA,
    ValueType& vA,
    KeyType& kB,
    ValueType& vB,
    bool dir,
    const CompFunc comp_t) {
  if (comp_t(kA, kB) == dir) {
    swap_var(kA, kB);
    swap_var(vA, vB);
  }
};

template <
    typename KeyType,
    typename ValueType,
    DPCPP::access::fence_space fence_space,
    typename CompFunc>
inline void bitonic_sort(
    const DPCPP::nd_item<1>& item,
    KeyType* key,
    ValueType* val,
    const CompFunc& comp_t,
    /* total sort problem size   */ const unsigned int sort_sz,
    /* stride between elements   */ const unsigned int stride = 1,
    /* adjust order              */ const bool inverse_order = false,
    /* min bitonic sequence size */ unsigned int min_seq_sz = 2) {
  auto item_id = item.get_local_id(0);
  auto local_sz = item.get_local_range(0);

  for (unsigned int ordered_seq_sz = min_seq_sz; ordered_seq_sz <= sort_sz;
       ordered_seq_sz *= 2) {
    for (unsigned int bitonic_seq_sz = ordered_seq_sz / 2; bitonic_seq_sz > 0;
         bitonic_seq_sz /= 2) {
      item.barrier(fence_space);
      for (unsigned int loc = item_id; loc < sort_sz / 2; loc += local_sz) {
        bool order =
            !(loc & (ordered_seq_sz >> 1)) && !(ordered_seq_sz == sort_sz);
        order = (ordered_seq_sz == sort_sz && inverse_order) ? !order : order;
        unsigned int pos_a = (2 * loc - (loc & (bitonic_seq_sz - 1))) * stride;
        unsigned int pos_b = pos_a + bitonic_seq_sz * stride;
        compare_and_swap(
            key[pos_a], val[pos_a], key[pos_b], val[pos_b], order, comp_t);
      }
    }
  }
  item.barrier(fence_space);
}

inline uint64_t last_power2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n < 16 ? 16 : n;
}

} // namespace impl

template <typename KeyType, typename ValueType, typename CompFunc>
void bitonic_merge_sort_kernel(
    KeyType* key,
    ValueType* val,
    const size_t sort_sz,
    const size_t outer_sz,
    const size_t inner_sz,
    const KeyType pad_k,
    const CompFunc comp_t) {
  auto& q = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto max_group_sz = dpcppMaxWorkGroupSize(dev_id);
  auto max_local_mem_sz = dpcppLocalMemSize(dev_id);

  // align size with bitonic sort requirement
  auto bitonic_sort_sz = impl::last_power2(sort_sz);
  auto local_sz =
      bitonic_sort_sz / 2 <= max_group_sz ? bitonic_sort_sz / 2 : max_group_sz;

  // slice by local memory usage
  // 1. put all into slm
  // 2. put local_sz * 2 into slm as a block
  //
  // assume local_sz is a number in power of 2,
  // then bitonic_blk_sort_sz must be a number in power of 2
  auto total_mem_sz = bitonic_sort_sz * (sizeof(KeyType) + sizeof(ValueType));
  auto bitonic_blk_sort_sz =
      total_mem_sz <= max_local_mem_sz ? bitonic_sort_sz : local_sz * 2;
  bool sliced = bitonic_blk_sort_sz != bitonic_sort_sz;

  auto g_key = key;
  auto g_val = val;

  // allocate padded global memory for global merge if needed
  KeyType* g_key_ = nullptr;
  ValueType* g_val_ = nullptr;
  at::Tensor key_tmp;
  at::Tensor val_tmp;
  if (sliced) {
    key_tmp = at::empty(
        bitonic_sort_sz * outer_sz * inner_sz * sizeof(KeyType),
        at::TensorOptions().device(kXPU).dtype(at::ScalarType::Byte));
    val_tmp = at::empty(
        bitonic_sort_sz * outer_sz * inner_sz * sizeof(ValueType),
        at::TensorOptions().device(kXPU).dtype(at::ScalarType::Byte));
    g_key_ = (KeyType*)key_tmp.data_ptr();
    g_val_ = (ValueType*)val_tmp.data_ptr();
  }

  auto cgf = DPCPP_Q_CGF(h) {
    auto s_key = dpcpp_local_acc_t<KeyType>(bitonic_blk_sort_sz, h);
    auto s_val = dpcpp_local_acc_t<ValueType>(bitonic_blk_sort_sz, h);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto item_id = item.get_local_id(0);
      auto batch = item.get_group(0);
      auto outer = batch / inner_sz;
      auto inner = batch % inner_sz;

      // batch_off: input global memory offset
      // batch_off_: temp global memory offset
      auto batch_off = outer * sort_sz * inner_sz + inner;
      // transpose if using temp global memory
      // (outer, sort, inner) -> (outer, inner, sort)
      // outer * bitonic_sort_sz * inner_sz + inner ->
      // outer * inner_sz * bitonic_sort_sz + inner * bitonic_sort_sz
      auto batch_off_ =
          outer * inner_sz * bitonic_sort_sz + inner * bitonic_sort_sz;

      // adjacent block is in inverse order for final global bitonic merge
      bool inverse_order = false;
      for (unsigned int blk = 0; blk < bitonic_sort_sz;
           blk += bitonic_blk_sort_sz) {
        auto blk_off = batch_off + blk * inner_sz;
        auto blk_off_ = batch_off_ + blk;

        if (/* must be a sliced case, (sliced && */ blk >= sort_sz) {
          for (auto loc = item_id; loc < bitonic_blk_sort_sz; loc += local_sz) {
            auto gbl_off_ = blk_off_ + loc;
            g_key_[gbl_off_] = pad_k;
            g_val_[gbl_off_] = 0;
          }
        } else {
          for (auto loc = item_id; loc < bitonic_blk_sort_sz; loc += local_sz) {
            auto loc_off = loc;
            auto gbl_off = blk_off + loc * inner_sz;
            s_key[loc_off] = (blk + loc < sort_sz) ? g_key[gbl_off] : pad_k;
            s_val[loc_off] = (blk + loc < sort_sz) ? g_val[gbl_off]
                                                   : static_cast<ValueType>(0);
          }

          impl::bitonic_sort<KeyType, ValueType, dpcpp_local_fence>(
              item,
              s_key.get_pointer().get(),
              s_val.get_pointer().get(),
              comp_t,
              bitonic_blk_sort_sz,
              /* stride */ 1,
              inverse_order);

          for (auto loc = item_id; loc < bitonic_blk_sort_sz; loc += local_sz) {
            auto loc_off = loc;
            if (sliced) {
              auto gbl_off_ = blk_off_ + loc;
              g_key_[gbl_off_] = s_key[loc_off];
              g_val_[gbl_off_] = s_val[loc_off];
            } else if (blk + loc < sort_sz) {
              auto gbl_off = blk_off + loc * inner_sz;
              g_key[gbl_off] = s_key[loc_off];
              g_val[gbl_off] = s_val[loc_off];
            }
          }

          inverse_order = !inverse_order;
        }
      }

      // global merge if needed
      if (sliced) {
        impl::bitonic_sort<KeyType, ValueType, dpcpp_global_fence>(
            item,
            g_key_ + batch_off_,
            g_val_ + batch_off_,
            comp_t,
            bitonic_sort_sz,
            /* transposed for contiguous */ 1,
            /* inverse_order */ false,
            bitonic_blk_sort_sz * 2);

        // transpose back
        for (auto loc = item_id; loc < sort_sz; loc += local_sz) {
          auto gbl_off = batch_off + loc * inner_sz;
          auto gbl_off_ = batch_off_ + loc;
          g_key[gbl_off] = g_key_[gbl_off_];
          g_val[gbl_off] = g_val_[gbl_off_];
        }
      }
    };

    h.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(outer_sz * inner_sz * local_sz),
            DPCPP::range<1>(local_sz)),
        kfn);
  };

  DPCPP_Q_SUBMIT(q, cgf);
}

// bubble sort for the first round sorting
template <typename KeyType, typename ValueType, typename CompFunc>
inline void leaf_sort(
    const DPCPP::item<1>& item,
    KeyType* key,
    ValueType* val,
    size_t n,
    size_t sorted_sz,
    const CompFunc& comp_t) {
  auto start = item.get_linear_id() * n;
  auto end = std::min(start + n, sorted_sz);
  for (size_t i = start; i < end; ++i) {
    for (size_t j = start + 1; j < start + end - i; ++j) {
      impl::compare_and_swap(
          key[j - 1], val[j - 1], key[j], val[j], false, comp_t);
    }
  }
}

// lower_bound used in merge sort: pick up the elements in the sequence doesn't
// meet the compare situation with smallest index
template <typename KeyType, typename CompFunc>
inline size_t lower_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (comp_t(in_data[it], key)) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <typename KeyType, typename CompFunc>
inline size_t upper_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (!comp_t(key, in_data[it])) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <typename KeyType, typename ValueType, typename CompFunc>
inline void merge(
    const size_t offset,
    KeyType* in_key,
    ValueType* in_val,
    KeyType* out_key,
    ValueType* out_val,
    const size_t sq1_start,
    const size_t sq1_end,
    const size_t sq2_start,
    const size_t sq2_end,
    const size_t chunk_size,
    const CompFunc& comp_t) {
  const size_t chunk1_start = std::min((offset + sq1_start), sq1_end);
  const size_t chunk1_end = std::min((chunk1_start + chunk_size), sq1_end);
  const size_t chunk2_start = std::min((offset + sq2_start), sq2_end);
  const size_t chunk2_end = std::min((chunk2_start + chunk_size), sq2_end);

  const size_t chunk1_size = chunk1_end - chunk1_start;
  const size_t chunk2_size = chunk2_end - chunk2_start;

  size_t l_sq2_low_bound;
  size_t r_sq2_low_bound;
  size_t l_sq1_upper_bound;
  size_t r_sq1_upper_bound;
  if (!comp_t(in_key[sq2_start], in_key[sq1_end - 1])) {
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[chunk1_start + i] = in_key[chunk1_start + i];
      out_val[chunk1_start + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[chunk2_start + i] = in_key[chunk2_start + i];
      out_val[chunk2_start + i] = in_val[chunk2_start + i];
    }
  } else if (!comp_t(in_key[sq1_start], in_key[sq2_end - 1])) {
    auto out1_offset = sq2_end - sq2_start + chunk1_start;
    auto out2_offset = sq1_start + chunk2_start - sq2_start;
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[out1_offset + i] = in_key[chunk1_start + i];
      out_val[out1_offset + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[out2_offset + i] = in_key[chunk2_start + i];
      out_val[out2_offset + i] = in_val[chunk2_start + i];
    }
  } else {
    // Process 1st sequence
    if (chunk1_start < chunk1_end) {
      const auto chunk1_l_item = in_key[chunk1_start];
      l_sq2_low_bound =
          lower_bound(in_key, sq2_start, sq2_end, chunk1_l_item, comp_t);
      const auto l_shift1 = chunk1_start - sq1_start;
      const auto l_shift2 = l_sq2_low_bound - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk1_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk1_start];
      if (chunk1_end - chunk1_start > 1) {
        const auto chunk1_r_item = in_key[chunk1_end - 1];
        r_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, sq2_end, chunk1_r_item, comp_t);
        const auto r_shift1 = chunk1_end - 1 - sq1_start;
        const auto r_shift2 = r_sq2_low_bound - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk1_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk1_end - 1];
      }
      for (auto idx = chunk1_start + 1; idx < chunk1_end - 1; ++idx) {
        const auto inter_item_1 = in_key[idx];
        l_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, r_sq2_low_bound, inter_item_1, comp_t);
        const auto shift1 = idx - sq1_start;
        const auto shift2 = l_sq2_low_bound - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_1;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
    // Process 2nd sequence
    if (chunk2_start < chunk2_end) {
      const auto chunk2_l_item = in_key[chunk2_start];
      l_sq1_upper_bound =
          upper_bound(in_key, sq1_start, sq1_end, chunk2_l_item, comp_t);
      const auto l_shift1 = l_sq1_upper_bound - sq1_start;
      const auto l_shift2 = chunk2_start - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk2_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk2_start];
      if (chunk2_end - chunk2_start > 1) {
        const auto chunk2_r_item = in_key[chunk2_end - 1];
        r_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, sq1_end, chunk2_r_item, comp_t);
        const auto r_shift1 = r_sq1_upper_bound - sq1_start;
        const auto r_shift2 = chunk2_end - 1 - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk2_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk2_end - 1];
      }

      for (auto idx = chunk2_start + 1; idx < chunk2_end - 1; ++idx) {
        const auto inter_item_2 = in_key[idx];
        l_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, r_sq1_upper_bound, inter_item_2, comp_t);
        const auto shift1 = l_sq1_upper_bound - sq1_start;
        const auto shift2 = idx - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_2;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
  }
}

template <int vec_size, typename KeyType, typename ValueType>
void vec_copy_kernel_impl(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  auto& q = dpcppGetCurrentQueue();
  using key_vec_t = at::native::Memory::aligned_vector_loop<KeyType, vec_size>;
  using val_vec_t =
      at::native::Memory::aligned_vector_loop<ValueType, vec_size>;
  key_vec_t* key_vec_ptr = reinterpret_cast<key_vec_t*>(key);
  key_vec_t* tmp_key_vec_ptr = reinterpret_cast<key_vec_t*>(tmp_key_data);
  val_vec_t* val_vec_ptr = reinterpret_cast<val_vec_t*>(val);
  val_vec_t* tmp_val_vec_ptr = reinterpret_cast<val_vec_t*>(tmp_val_data);
  auto num_work_item = CeilDiv(sort_sz, (size_t)vec_size);
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto item_id = item.get_linear_id();
      int remaining = sort_sz - item_id * vec_size;
      if (remaining < vec_size) {
        for (int index = 0; index < remaining; index++) {
          auto offset = item_id * vec_size + index;
          key[offset] = tmp_key_data[offset];
          val[offset] = tmp_val_data[offset];
        }
      } else {
#pragma unroll
        for (int index = 0; index < vec_size; index++) {
          key_vec_ptr[item_id][index] = tmp_key_vec_ptr[item_id][index];
          val_vec_ptr[item_id][index] = tmp_val_vec_ptr[item_id][index];
        }
      }
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(num_work_item), kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

template <typename KeyType, typename ValueType>
void copy_to_dst(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  int vec_size_key = at::native::Memory::can_vectorize_up_to_loop<KeyType>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(key));
  auto vec_size_val = at::native::Memory::can_vectorize_up_to_loop<ValueType>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(val));
  auto vec_size = std::min(vec_size_key, vec_size_val);

#define VEC_COPY_KERNEL_IMPL(vec_size)                  \
  {                                                     \
    vec_copy_kernel_impl<vec_size, KeyType, KeyType>(   \
        key, tmp_key_data, val, tmp_val_data, sort_sz); \
  }

  switch (vec_size) {
    case 8: {
      VEC_COPY_KERNEL_IMPL(8);
      break;
    }
    case 4: {
      VEC_COPY_KERNEL_IMPL(4);
      break;
    }
    case 2: {
      VEC_COPY_KERNEL_IMPL(2);
      break;
    }
    case 1: {
      VEC_COPY_KERNEL_IMPL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for merge sort. vec size ",
          vec_size);
  }
#undef VEC_COPY_KERNEL_IMPL
}

// merge sort: only for 1d (single batch) tensor sort
template <typename KeyType, typename ValueType, typename CompFunc>
void merge_sort_kernel(
    KeyType* key,
    ValueType* val,
    const size_t sort_sz,
    const CompFunc comp_t) {
  RECORD_FUNCTION("merge_sort_kernel", {});
  const size_t leaf = 4;
  const size_t optimal_chunk = 4;

  const size_t leaf_step = ((sort_sz - 1) / leaf) + 1;
  auto& q = dpcppGetCurrentQueue();
  // 1, leaf sort
  auto cgf_1 = DPCPP_Q_CGF(h) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      leaf_sort<KeyType, ValueType>(item, key, val, leaf, sort_sz, comp_t);
    };
    h.parallel_for(DPCPP::range<1>(leaf_step), kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf_1);

  auto key_options = map_options<KeyType>();
  auto val_options = map_options<ValueType>();
  Tensor tmp_key = at::empty({sort_sz}, key_options);
  Tensor tmp_val = at::empty({sort_sz}, val_options);
  auto tmp_key_data = tmp_key.data_ptr<KeyType>();
  auto tmp_val_data = tmp_val.data_ptr<ValueType>();

  bool data_in_tmp = false;

  size_t sorted = leaf;
  size_t chunk = std::min(leaf, optimal_chunk);

  while (sorted < sort_sz) {
    size_t sorted_pair = 2 * sorted;
    size_t chunk_num_per_sorted = sorted / chunk;
    size_t full_pairs = sort_sz / sorted_pair;
    size_t incomplete_pair = sort_sz - sorted_pair * full_pairs;
    size_t first_block_in_incomplete_pair =
        incomplete_pair > sorted ? sorted : incomplete_pair;
    size_t incomplete_last_chunk = first_block_in_incomplete_pair % chunk != 0;
    size_t incomplete_pair_steps =
        first_block_in_incomplete_pair / chunk + incomplete_last_chunk;
    size_t full_pair_steps = full_pairs * chunk_num_per_sorted;
    size_t steps = full_pair_steps + incomplete_pair_steps;

    auto cgf_2 = DPCPP_Q_CGF(h) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
        const size_t idx = item.get_linear_id();
        const size_t sq1_start =
            std::min(sorted_pair * ((idx * chunk) / sorted), sort_sz);
        const size_t sq1_end = std::min(sq1_start + sorted, sort_sz);
        const size_t sq2_start = sq1_end;
        const size_t sq2_end = std::min(sq2_start + sorted, sort_sz);

        const size_t offset_in_sq = chunk * (idx % chunk_num_per_sorted);

        if (!data_in_tmp) {
          merge(
              offset_in_sq,
              key,
              val,
              tmp_key_data,
              tmp_val_data,
              sq1_start,
              sq1_end,
              sq2_start,
              sq2_end,
              chunk,
              comp_t);
        } else {
          merge(
              offset_in_sq,
              tmp_key_data,
              tmp_val_data,
              key,
              val,
              sq1_start,
              sq1_end,
              sq2_start,
              sq2_end,
              chunk,
              comp_t);
        }
      };
      h.parallel_for(DPCPP::range<1>(steps), kfn);
    };
    DPCPP_Q_SUBMIT(q, cgf_2);

    data_in_tmp = !data_in_tmp;
    sorted = sorted_pair;
    if (chunk < optimal_chunk)
      chunk *= 2;
  }
  if (data_in_tmp) {
    copy_to_dst<KeyType, ValueType>(
        key, tmp_key_data, val, tmp_val_data, sort_sz);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
