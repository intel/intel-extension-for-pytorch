#pragma once

#include <ATen/ATen.h>
#include <utils/DPCPP.h>

#include "comm/General.h"
#include "comm/KeyTraits.h"
#include "comm/TensorOptions.h"

#include "SortingRadixProcesser.h"

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename KeyT,
    typename ValueT,
    int GROUP_ITEMS,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT,
    int SUBGROUP_SIZE,
    int KEYS_PER_ITEM,
    typename SelectMethod,
    typename KeyTraitsT>
struct FastGroupRadixSelectImplFunctor {
  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    auto slice = item.get_group_linear_id();
    auto lid = item.get_local_id(0);

    auto offset_s = slice * nelements;
    auto key_in_begin = key_in + offset_s;
    auto value_in_begin = value_in + offset_s;
    auto key_out_begin = key_out + slice * ntopk;
    auto value_out_begin = value_out + slice * ntopk;

    auto method = SelectMethod(item, slm);

    KeyT keys[SelectMethod::REG_LEN];
    PrivateValueT values[SelectMethod::REG_LEN];

    KeyT* keys_temp = reinterpret_cast<KeyT*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get() +
        SelectMethod::GetSharedLocalMemorySize());
    PrivateValueT* values_temp = reinterpret_cast<PrivateValueT*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get() +
        SelectMethod::GetSharedLocalMemorySize() + ntopk * sizeof(KeyT));

#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
      int offset = lid * KEYS_PER_ITEM + ITEM;
      if (offset < nelements) {
        keys[ITEM] = key_in_begin[offset];
        values[ITEM] = USE_INDICES ? offset : value_in_begin[offset];
      } else {
        keys[ITEM] = padding_key;
      }
    }

    int num_start = SelectMethod::PROCESSING_LENGTH;
    while (num_start < nelements) {
      method.select_group(
          keys, values, sizeof(KeyT) * 8, 0, ntopk, keys_temp, values_temp);
      item.barrier(dpcpp_local_fence);
#pragma unroll
      for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
        int offset = lid * KEYS_PER_ITEM + ITEM;
        if (offset < ntopk) {
          keys[ITEM] = keys_temp[offset];
          values[ITEM] = values_temp[offset];
        } else {
          offset += num_start - ntopk;
          if (offset < nelements) {
            keys[ITEM] = key_in_begin[offset];
            values[ITEM] = USE_INDICES ? offset : value_in_begin[offset];
          } else {
            keys[ITEM] = padding_key;
          }
        }
      }
      num_start += SelectMethod::PROCESSING_LENGTH - ntopk;
      item.barrier(dpcpp_local_fence);
    }

    method.select_group(
        keys,
        values,
        sizeof(KeyT) * 8,
        0,
        ntopk,
        key_out_begin,
        value_out_begin);
  }
  FastGroupRadixSelectImplFunctor(
      const KeyT* key_in_,
      KeyT* key_out_,
      const ValueT* value_in_,
      ValueT* value_out_,
      int nsegments_,
      int nelements_,
      int ntopk_,
      dpcpp_local_acc_t<unsigned char> slm_,
      KeyT padding_key_)
      : key_in(key_in_),
        key_out(key_out_),
        value_in(value_in_),
        value_out(value_out_),
        nsegments(nsegments_),
        nelements(nelements_),
        ntopk(ntopk_),
        slm(slm_),
        padding_key(padding_key_) {}

 private:
  const KeyT* key_in;
  KeyT* key_out;
  const ValueT* value_in;
  ValueT* value_out;
  int nsegments;
  int nelements;
  int ntopk;
  dpcpp_local_acc_t<unsigned char> slm;
  KeyT padding_key;
};

template <
    typename KeyT,
    typename ValueT,
    int GROUP_ITEMS,
    bool IS_DESCENDING,
    bool USE_INDICES,
    typename PrivateValueT = ValueT,
    int SUBGROUP_SIZE>
inline void fast_group_radix_select_impl_(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    int nsegments,
    int nelements,
    int ntopk) {
  constexpr int KEYS_PER_ITEM = 4;

  using SelectMethod = GroupRadixProcesser<
      KeyT,
      GROUP_ITEMS,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      PrivateValueT>;

  using KeyTraitsT = typename SelectMethod::KeyTraitsT;
  auto& q = dpcppGetCurrentQueue();
  auto padding_key = IS_DESCENDING ? Numerics<KeyT>::lower_bound()
                                   : Numerics<KeyT>::upper_bound();

  auto cgf = DPCPP_Q_CGF(h) {
    auto slm = dpcpp_local_acc_t<unsigned char>(
        SelectMethod::GetSharedLocalMemorySize() +
            ntopk * (sizeof(KeyT) + sizeof(ValueT)),
        h);
    FastGroupRadixSelectImplFunctor<
        KeyT,
        ValueT,
        GROUP_ITEMS,
        IS_DESCENDING,
        USE_INDICES,
        PrivateValueT,
        SUBGROUP_SIZE,
        KEYS_PER_ITEM,
        SelectMethod,
        KeyTraitsT>
        kfn(key_in,
            key_out,
            value_in,
            value_out,
            nsegments,
            nelements,
            ntopk,
            slm,
            padding_key);
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(nsegments * GROUP_ITEMS),
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
inline void fast_group_radix_select_impl(
    const KeyT* key_in,
    KeyT* key_out,
    const ValueT* value_in,
    ValueT* value_out,
    int nsegments,
    int nelements,
    int ntopk) {
  auto* dev_prop = dpcppGetDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  switch (dev_prop->subgroup_sizes[0] * 2) {
    // TODO: Fixed subgroup size is used currently for performance consideration
    // however, runtime acquisition is better for scalability
    case 32:
      fast_group_radix_select_impl_<
          KeyT,
          ValueT,
          GROUP_ITEMS,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          32>(
          key_in, key_out, value_in, value_out, nsegments, nelements, ntopk);
      break;
    default:
      fast_group_radix_select_impl_<
          KeyT,
          ValueT,
          GROUP_ITEMS,
          IS_DESCENDING,
          USE_INDICES,
          PrivateValueT,
          16>(
          key_in, key_out, value_in, value_out, nsegments, nelements, ntopk);
      break;
  }
}

inline uint64_t radix_select_last_power2(uint64_t n) {
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
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    bool USE_INDICES>
inline void fast_group_radix_select_(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int num_topk) {
  auto max_group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
#define DISPATCH_SELECT_IMPL(PADDING_NUM_ELEMENTS) \
  {                                                \
    fast_group_radix_select_impl<                  \
        key_t,                                     \
        value_t,                                   \
        PADDING_NUM_ELEMENTS / 4,                  \
        IS_DESCENDING,                             \
        USE_INDICES>(                              \
        keys_in,                                   \
        keys_out,                                  \
        values_in,                                 \
        values_out,                                \
        num_segments,                              \
        num_elements,                              \
        num_topk);                                 \
  }
  if (num_elements <= max_group_size * 4) {
    switch (radix_select_last_power2(num_elements)) {
      case 4096:
        DISPATCH_SELECT_IMPL(4096);
        break;
      case 2048:
        DISPATCH_SELECT_IMPL(2048);
        break;
      case 1024:
        DISPATCH_SELECT_IMPL(1024);
        break;
      case 512:
        DISPATCH_SELECT_IMPL(512);
        break;
      default:
        DISPATCH_SELECT_IMPL(256);
        break;
    }
  } else {
    switch (max_group_size) {
      case 1024:
        DISPATCH_SELECT_IMPL(4096);
        break;
      case 512:
        DISPATCH_SELECT_IMPL(2048);
        break;
      default:
        DISPATCH_SELECT_IMPL(1024);
        break;
    }
  }
#undef DISPATCH_SELECT_IMPL
}

template <typename key_t, typename value_t>
void fast_group_select_pairs(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int num_topk,
    bool is_descending) {
#define DISPATH_INDICE(X)                                 \
  {                                                       \
    if (is_descending)                                    \
      fast_group_radix_select_<key_t, value_t, true, X>(  \
          keys_in,                                        \
          keys_out,                                       \
          values_in,                                      \
          values_out,                                     \
          num_segments,                                   \
          num_elements,                                   \
          num_topk);                                      \
    else                                                  \
      fast_group_radix_select_<key_t, value_t, false, X>( \
          keys_in,                                        \
          keys_out,                                       \
          values_in,                                      \
          values_out,                                     \
          num_segments,                                   \
          num_elements,                                   \
          num_topk);                                      \
  }
  if (values_in == nullptr) {
    DISPATH_INDICE(true);
  } else {
    DISPATH_INDICE(false);
  }
#undef DISPATH_INDICE
}

} // namespace AtenIpexTypeXPU
} // namespace at
