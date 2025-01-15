// Porting from ipex
// will upstream to pytorch when in tree
#pragma once

// #include <comm/xpu_aten.h>
#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DimVector.h>
#include <ATen/Dispatch.h>
#include <ATen/Formatting.h>
// #include <ATen/Functions.h>
#include <ATen/NamedTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/Version.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/UnsafeFromTH.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Allocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include <ATen/CPUApplyUtils.h>

#include <ATen/native/xpu/sycl/IntegerDivider.h>

namespace at {
namespace xpu {
namespace detail {

#define XPU_MAX_TENSORINFO_DIMS 12

template <typename T, typename IndexType>
struct TensorInfo {
  using scalar_t = T;

  TensorInfo();
  TensorInfo(
      T* p,
      int dim,
      IndexType sz[XPU_MAX_TENSORINFO_DIMS],
      IndexType st[XPU_MAX_TENSORINFO_DIMS]);

  // Set the size of given dimension to 1, as if were a
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

  T* data = nullptr;
  IndexType sizes[XPU_MAX_TENSORINFO_DIMS];
  IndexType strides[XPU_MAX_TENSORINFO_DIMS];
  int dims = 0;
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
    IndexType sz[XPU_MAX_TENSORINFO_DIMS],
    IndexType st[XPU_MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  TORCH_INTERNAL_ASSERT(dims <= XPU_MAX_TENSORINFO_DIMS);

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
template <typename T, typename IndexType, bool Trivial = false>
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

    for (int dim = info.dims - 1; dim > 0; --dim) {
      IndexType curDimIndex = linearId % info.sizes[dim];
      IndexType curDimOffset = curDimIndex * info.strides[dim];
      offset += curDimOffset;
      linearId /= info.sizes[dim];
    }
    return offset + linearId * info.strides[0];
  }
};

// To isolate unnecessary code, even the code is not involved in
// contiguouse case. Additional unnecessary code impacts efficiency of
// generated code.
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, true> {
  static constexpr bool STRICT_CONTIGUOUS = true;
  static constexpr bool NON_STRICT_CONTIGUOUS = false;
  static inline IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info,
      bool strict_contiguous = true) {
    return linearId;
  }
};

template <typename scalar, typename IndexType>
TensorInfo<scalar, IndexType> getTensorInfo(const at::TensorBase& t) {
  IndexType sz[XPU_MAX_TENSORINFO_DIMS];
  IndexType st[XPU_MAX_TENSORINFO_DIMS];

  TORCH_CHECK(
      t.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "dim:",
      t.dim(),
      " exceed max allowed dim:",
      XPU_MAX_TENSORINFO_DIMS);

  int dims;
  if (t.dim()) {
    dims = t.dim();
    for (int i = 0; i < dims; ++i) {
      sz[i] = t.size(i);
      st[i] = t.stride(i);
    }
  } else {
    dims = 1;
    sz[0] = 1;
    st[0] = 1;
  }

  scalar* data_ptr = nullptr;

  if constexpr (std::is_const<scalar>::value) {
    data_ptr = t.const_data_ptr<scalar>();
  } else {
    data_ptr = t.mutable_data_ptr<scalar>();
  }

  return TensorInfo<scalar, IndexType>(data_ptr, dims, sz, st);
}

} // namespace detail
} // namespace xpu
} // namespace at
