#pragma once

#include <c10/macros/Macros.h>
#include <array>
#include <cstdint>

#include <core/Array.h>
#include <core/IntegerDivider.h>

/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
/// operands that share the same shape, but may have different strides.

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 12;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = xpu::dpcpp::Array<index_t, NARGS>;

  OffsetCalculator(
      int dims,
      const int64_t* sizes,
      const int64_t* const* strides,
      const int64_t* element_sizes = nullptr)
      : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>25) dims");
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < dims) {
        sizes_[dim] = IntDivider<index_t>(sizes[dim]);
      } else {
        sizes_[dim] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[dim][arg] = dim < dims ? strides[arg][dim] / element_size : 0;
      }
    }
  }

  offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < dims) {
        auto divmod = sizes_[dim].divmod(linear_idx);
        linear_idx = divmod.div;

#pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
          offsets[arg] += divmod.mod * strides_[dim][arg];
        }
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][NARGS];
};

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // The offsets are in # of elements, not in bytes.
  using offset_type = xpu::dpcpp::Array<index_t, std::max<int>(NARGS, 1)>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};
