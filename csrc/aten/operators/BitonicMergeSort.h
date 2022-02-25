#pragma once

#include <ATen/ATen.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/Numerics.h"

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
    DPCPP::memory_scope fence_scope,
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
      DPCPP::group_barrier(item.get_group(), fence_scope);
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
  DPCPP::group_barrier(item.get_group(), fence_scope);
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

          impl::bitonic_sort<KeyType, ValueType, dpcpp_mem_scp_wg>(
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
        impl::bitonic_sort<KeyType, ValueType, dpcpp_mem_scp_dev>(
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

} // namespace AtenIpexTypeXPU
} // namespace at
