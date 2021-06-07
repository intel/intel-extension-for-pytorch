#pragma once

#include <array>
#include <cstdint>
#include <c10/macros/Macros.h>

#include <operators/comm/Array.h>
#include <utils/IntegerDivider.h>


/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
/// operands that share the same shape, but may have different strides.

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 12;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = xpu::dpcpp::Array<index_t, NARGS>;

  // This is a workaround for the compute cpp.
  // An issue was found that if the tailing member data is the two dim array
  // and the column number is larger than 1, the dpcpp kernel may write the memory
  // unexpectedly, memory leakage happened.
  // These test cases have been tried.
  // index_t strides_[MAX_DIMS][NARGS]; Error
  // index_t strides_[2][2]; Error
  // index_t strides_[2][1]; Ok
  // index_t strides_[MAX_DIMS*NARGS][1]; Ok
  // The reduce and sum kernel access the strides_[0][0]. So the workaround is as it is.
  static inline int __wa(int dim, int arg) {
    return NARGS * dim + arg;
  }

  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>25) dims");
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[__wa(i, arg)][0] = i < dims ? strides[arg][i] / element_size : 0;
      }
    }
  }

  offset_type get(index_t linear_idx) const {
    offset_type offsets;
    // #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    // #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      // #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[__wa(dim, arg)][0];
      }

    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS*NARGS][1];
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
