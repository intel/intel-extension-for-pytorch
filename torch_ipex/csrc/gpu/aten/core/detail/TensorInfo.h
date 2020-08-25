#pragma once

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>

namespace at {
namespace dpcpp {
namespace detail {

#ifdef USE_USM
#define MAX_TENSORINFO_DIMS 12
#else
// Setting to 5 is to work-around dpcpp kernel argument limitation (1024B).
#define MAX_TENSORINFO_DIMS 5
#endif

#define MAX_DPCPPTORCH_DIMS MAX_TENSORINFO_DIMS

#define DPCPPTORCH_STR(X) #X
#define DPCPPTORCH_DIM_WARNING                      \
  "tensor too large or too many (>" DPCPPTORCH_STR( \
      MAX_DPCPPTORCH_DIMS) ") dimensions"

// DPCPP kernel argument taht defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo();
  TensorInfo(
      T* p,
      int dim,
      IndexType sz[MAX_TENSORINFO_DIMS],
      IndexType st[MAX_TENSORINFO_DIMS]);

  // Set the sive of given dimension to 1, as if were a
  // reduction dim (allow you to calculate offsets of the
  // reduction slice)
  void reduceDim(int dim);

  // See note on [collapse dims].
  int collapseDims(const int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo() {
  data = nullptr;
  dims = 0;
}

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(
    T* p,
    int dim,
    IndexType sz[MAX_TENSORINFO_DIMS],
    IndexType st[MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  TORCH_INTERNAL_ASSERT(dims <= MAX_TENSORINFO_DIMS);

  for (int i = 0; i < dim; i++) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename T, typename IndexType>
void TensorInfo<T, IndexType>::reduceDim(int dim) {
  TORCH_CHECK(dim < dims && dim >= 0, "expect dim between 0 and dims - 1");
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {
  auto result = at::collapse_dims(sizes, strides, dims, excludeDim);
  dims = std::get<1>(result);
  return std::get<0>(result);
}

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims = -1>
struct IndexToOffset {
  static IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;

    // Uses static dims
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;

    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

// OffsetInfo is a faster implementation of IndexToOffset that uses faster
// integer division: we transform each division into integer multiplication by a
// pre-computed constant.  (See IntDivider for details.)
template <typename T, typename IndexType, int Dims>
struct OffsetInfo {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo) {
    assert(tinfo.dims == Dims);
    data = tinfo.data;

    for (int i = 0; i < Dims; ++i) {
      sizes[i] = tinfo.sizes[i];
      strides[i] = tinfo.strides[i];
    }
  }

  T* get(IndexType linearIndex) const {
    IndexType offset = 0;

    for (int i = Dims - 1; i > 0; --i) {
      linearIndex = sizes[i] / linearIndex;
      offset += (sizes[i] % linearIndex) * strides[i];
    }

    return &data[offset + linearIndex * strides[0]];
  }

  T* data;
  IndexType sizes[Dims];
  IndexType strides[Dims];
};

// For 1D tensors the offset equals linear index * stride.
template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, 1> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
      : data{tinfo.data}, stride{tinfo.strides[0]} {}

  T* get(IndexType linearIndex) const {
    return &data[linearIndex * stride];
  }

  T* data;
  const IndexType stride;
};

// Dims=-1 is used when the dimension is unknown at compile time.
//
// Unfortunately, pre-computation does not work here.
// So let's fall back to vanilla division approach.

template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -1> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo) : tinfo(tinfo) {}

  T* get(IndexType linearIndex) const {
    IndexType offset = IndexToOffset<T, IndexType, -1>::get(linearIndex, tinfo);
    return &tinfo.data[offset];
  }

  TensorInfo<T, IndexType> tinfo;
};

template <typename scalar, typename IndexType>
TensorInfo<scalar, IndexType> getTensorInfo(const at::Tensor& t) {
  IndexType sz[MAX_TENSORINFO_DIMS];
  IndexType st[MAX_TENSORINFO_DIMS];

  int dims = t.dim();
  for (int i = 0; i < dims; ++i) {
    sz[i] = t.size(i);
    st[i] = t.stride(i);
  }

  return TensorInfo<scalar, IndexType>(t.data_ptr<scalar>(), dims, sz, st);
}

} // detail
} // dpcpp
} // at
