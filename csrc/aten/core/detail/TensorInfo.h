#pragma once

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>

#include <core/IntegerDivider.h>

namespace xpu {
namespace dpcpp {
namespace detail {

#define MAX_TENSORINFO_DIMS 12

#define MAX_DPCPPTORCH_DIMS MAX_TENSORINFO_DIMS

#define DPCPPTORCH_STR(X) #X
#define DPCPPTORCH_DIM_WARNING                      \
  "tensor too large or too many (>" DPCPPTORCH_STR( \
      MAX_DPCPPTORCH_DIMS) ") dimensions"

// DPCPP kernel argument taht defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  using scalar_t = T;

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

  int outerSize(const int dim);

  int innerSize(const int dim);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  inline bool isContiguous() const {
    return dims == 1 && strides[0] == 1;
  }

  inline bool isContiguousCheckStrict(bool strict_contiguous) const {
    if (strict_contiguous)
      return is_strict_contiguous;
    else
      return is_contiguous;
  }

  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;
  bool is_contiguous;
  bool is_strict_contiguous;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo() {
  data = nullptr;
  dims = 0;
  is_contiguous = true;
  is_strict_contiguous = true;
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

  is_contiguous = true;
  int z = 1;
  for (int i = dim - 1; i >= 0; i--) {
    sizes[i] = sz[i];
    strides[i] = st[i];

    if (is_contiguous && strides[i] == z) {
      z *= sizes[i];
    } else {
      is_contiguous = false;
    }
  }

  is_strict_contiguous = dims == 1 && strides[0] == 1;
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

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::innerSize(const int exclusive) {
  int size = 1;
  for (int i = dims - 1; i > exclusive; i--) {
    size *= sizes[i];
  }
  return size;
}

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::outerSize(const int exclusive) {
  int size = 1;
  for (int i = 0; i < exclusive; i++) {
    size *= sizes[i];
  }
  return size;
}

// Translate a linear index for the apply to a T* offset;
template <typename T, typename IndexType, /* deprecated */ int DIM = -1>
struct IndexToOffset {
  static constexpr bool STRICT_CONTIGUOUS = true;
  static constexpr bool NON_STRICT_CONTIGUOUS = false;
  static inline IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info,
      bool strict_contiguous = true) {
    IndexType offset = 0;

    if (info.isContiguousCheckStrict(strict_contiguous)) {
      return linearId;
    }

#pragma unroll
    for (int dim = MAX_TENSORINFO_DIMS - 1; dim >= 0; --dim) {
      if (dim < info.dims) {
        auto divider = IntDivider<IndexType>(info.sizes[dim]);
        auto divmod = divider.divmod(linearId);
        linearId = divmod.div;
        offset += divmod.mod * info.strides[dim];
      }
    }
    return offset;
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

} // namespace detail
} // namespace dpcpp
} // namespace xpu
