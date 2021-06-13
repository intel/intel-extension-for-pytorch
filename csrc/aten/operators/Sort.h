#pragma once

#include <ATen/ATen.h>

#include <runtime/DPCPP.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include "comm/Numerics.h"

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

// Collection of kernel sort routimes
// Collection of kernel sort routines
template <typename T>
struct LTComp {
  inline bool operator()(const T& a, const T& b) const {
    return Numerics<T>::lt(a, b);
  }
};

template <typename T>
struct GTComp {
  inline bool operator()(const T& a, const T& b) const {
    return Numerics<T>::gt(a, b);
  }
};

template <typename T>
inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
inline void bitonicSwap(
    K& kA,
    V& vA,
    bool& validA,
    K& kB,
    V& vB,
    bool& validB,
    bool dir,
    const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};

template <
    typename Comparator,
    typename K,
    typename V,
    typename IndexType,
    int Power2SortSize>
inline void bitonicSort(
    const dpcpp_local_acc_t<K>& keys_smem,
    const dpcpp_local_acc_t<V>& values_smem,
    const dpcpp_local_acc_t<bool>& valid_smem,
    const Comparator& comp,
    const DPCPP::nd_item<1>& item_id) {
  auto thread_id = item_id.get_local_id(0);
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((thread_id & (size / 2)) != 0);

    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      item_id.barrier(dpcpp_local_fence);
      unsigned int pos = 2 * thread_id - (thread_id & (stride - 1));
      bitonicSwap<Comparator, K, V>(
          keys_smem[pos],
          values_smem[pos],
          valid_smem[pos],
          keys_smem[pos + stride],
          values_smem[pos + stride],
          valid_smem[pos + stride],
          flag,
          comp);
    }
  }

  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    item_id.barrier(dpcpp_local_fence);
    unsigned int pos = 2 * thread_id - (thread_id & (stride - 1));
    bitonicSwap<Comparator, K, V>(
        keys_smem[pos],
        values_smem[pos],
        valid_smem[pos],
        keys_smem[pos + stride],
        values_smem[pos + stride],
        valid_smem[pos + stride],
        false,
        comp);
  }

  item_id.barrier(dpcpp_local_fence);
}

template <
    typename K,
    typename V,
    int KeyDims,
    int ValueDims,
    typename Comparator,
    typename IndexType,
    int Power2SortSize>
class binarySortKVInplaceKernelName {};

template <
    typename K,
    typename V,
    int KeyDims,
    int ValueDims,
    typename Comparator,
    typename IndexType,
    int Power2SortSize>
inline void bitonicSortKVInPlace(
    TensorInfo<K, IndexType> keys,
    IndexType keySlices,
    IndexType keySliceSize,
    IndexType keySliceStride,
    TensorInfo<V, IndexType> values,
    IndexType valueSliceStride,
    Comparator comp) {
  // Find the slice of the tensor that we are sorting
  auto queue = dpcppGetCurrentQueue();
  int64_t local_size = Power2SortSize / 2;
  int64_t global_size = keySlices * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto keys_data = get_buffer<dpcpp_rw_mode>(cgh, keys.data);
    auto values_data = get_buffer<dpcpp_rw_mode>(cgh, values.data);
    auto sharedKeys_acc = dpcpp_local_acc_t<K>(Power2SortSize, cgh);
    auto sharedValues_acc = dpcpp_local_acc_t<V>(Power2SortSize, cgh);
    auto sharedValid_acc = dpcpp_local_acc_t<bool>(Power2SortSize, cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto thread_id = item_id.get_local_id(0);
      auto group_id = item_id.get_group_linear_id();
      const IndexType keyStartOffset =
          IndexToOffset<K, IndexType, KeyDims>::get(group_id, keys);
      const IndexType valueStartOffset =
          IndexToOffset<V, IndexType, ValueDims>::get(group_id, values);
      auto keys_ptr = get_pointer(keys_data) + keyStartOffset;
      auto values_ptr = get_pointer(values_data) + valueStartOffset;
      // If the sort size is 1, the data is already sorted
      if (Power2SortSize == 1) {
        return;
      }
      // Otherwise, each thread is responsible for loading and storing 2
      // elements. The sort size is guaranteed to be >= 2
      const int elem1 = thread_id;
      const int elem2 = thread_id + (Power2SortSize / 2);

      bool valid1 = (static_cast<IndexType>(elem1) < keySliceSize);
      K k1 = valid1 ? keys_ptr[elem1 * keySliceStride]
                    : ScalarConvert<int, K>::to(0);
      V v1 = valid1 ? values_ptr[elem1 * valueSliceStride]
                    : ScalarConvert<int, V>::to(0);
      sharedKeys_acc[elem1] = k1;
      sharedValues_acc[elem1] = v1;
      sharedValid_acc[elem1] = valid1;
      bool valid2 = (static_cast<IndexType>(elem2) < keySliceSize);
      K k2 = valid2 ? keys_ptr[elem2 * keySliceStride]
                    : ScalarConvert<int, K>::to(0);
      V v2 = valid2 ? values_ptr[elem2 * valueSliceStride]
                    : ScalarConvert<int, V>::to(0);
      sharedKeys_acc[elem2] = k2;
      sharedValues_acc[elem2] = v2;
      sharedValid_acc[elem2] = valid2;
      // Sort!
      bitonicSort<Comparator, K, V, IndexType, Power2SortSize>(
          sharedKeys_acc, sharedValues_acc, sharedValid_acc, comp, item_id);
      // elem1 and elem2 values might be out-of-range, if the data size we are
      // sorting is smaller than half the power2 size
      if (valid1) {
        keys_ptr[elem1 * keySliceStride] = sharedKeys_acc[elem1];
        values_ptr[elem1 * valueSliceStride] = sharedValues_acc[elem1];
      }
      if (valid2) {
        keys_ptr[elem2 * keySliceStride] = sharedKeys_acc[elem2];
        values_ptr[elem2 * valueSliceStride] = sharedValues_acc[elem2];
      }
    };

    cgh.parallel_for<binarySortKVInplaceKernelName<
        K,
        V,
        KeyDims,
        ValueDims,
        Comparator,
        IndexType,
        Power2SortSize>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
static inline uint64_t nextHighestPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
template <typename scalar_t>
inline void SortKeyValueInplace(Tensor& key, Tensor& value, int dim, bool dir) {
  TORCH_CHECK(
      key.sizes().equals(value.sizes()),
      "Key tensor must have same size as value tensor");
  int dims = value.dim() == 0 ? 1 : value.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  dims = key.dim() == 0 ? 1 : key.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  ptrdiff_t inElements = key.numel();

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.dim() == 0 ? 1 : key.size(dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  if (ceilPowerOf2 > 2048) {
    TORCH_CHECK(
        false, "sortKeyValueInplace only works for sizes <= 2048 at present");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                   \
  do {                                               \
    int blockSize = SIZE / 2;                        \
    if (blockSize < 1) {                             \
      blockSize = 1;                                 \
    }                                                \
                                                     \
    if (dir) {                                       \
      bitonicSortKVInPlace<                          \
          scalar_t,                                  \
          int64_t,                                   \
          A,                                         \
          -1,                                        \
          GTComp<scalar_t>,                          \
          TYPE,                                      \
          SIZE>(                                     \
          keyInfo,                                   \
          keySlices,                                 \
          (TYPE)keySliceSize,                        \
          (TYPE)keyInfo.strides[collapseKeyDim],     \
          valueInfo,                                 \
          (TYPE)valueInfo.strides[collapseValueDim], \
          GTComp<scalar_t>());                       \
    } else {                                         \
      bitonicSortKVInPlace<                          \
          scalar_t,                                  \
          int64_t,                                   \
          A,                                         \
          -1,                                        \
          LTComp<scalar_t>,                          \
          TYPE,                                      \
          SIZE>(                                     \
          keyInfo,                                   \
          keySlices,                                 \
          (TYPE)keySliceSize,                        \
          (TYPE)keyInfo.strides[collapseKeyDim],     \
          valueInfo,                                 \
          (TYPE)valueInfo.strides[collapseValueDim], \
          LTComp<scalar_t>());                       \
    }                                                \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                \
  {                                              \
    switch (ceilPowerOf2) {                      \
      case 2048:                                 \
      case 1024:                                 \
      case 512:                                  \
      case 256:                                  \
        HANDLE_CASE(TYPE, A, 512);               \
        break;                                   \
      case 128:                                  \
      case 64:                                   \
        HANDLE_CASE(TYPE, A, 128);               \
        break;                                   \
      case 32:                                   \
      case 16:                                   \
      case 8:                                    \
      case 4:                                    \
      case 2:                                    \
        HANDLE_CASE(TYPE, A, 32);                \
        break;                                   \
      case 1:                                    \
        /* Nothing to do, data already sorted */ \
        break;                                   \
      default:                                   \
        assert(false);                           \
    }                                            \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (canUse32BitIndexMath(key)) {
    TensorInfo<scalar_t, unsigned int> keyInfo =
        getTensorInfo<scalar_t, unsigned int>(key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, unsigned int> valueInfo =
        getTensorInfo<int64_t, unsigned int>(value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    TensorInfo<scalar_t, uint64_t> keyInfo =
        getTensorInfo<scalar_t, uint64_t>(key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, uint64_t> valueInfo =
        getTensorInfo<int64_t, uint64_t>(value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    HANDLE_SORT_CASE(uint64_t, -1);
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE
}
