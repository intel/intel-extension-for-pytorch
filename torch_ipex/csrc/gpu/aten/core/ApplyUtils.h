#pragma once

#include <ATen/TensorUtils.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>

#include <core/Context.h>

#include <math.h>

namespace at {
namespace dpcpp {

// We pull the kernel name from anonymous namespace to outside,
// because otherwise dpcpp compiler will fail to recognize
// these kernel name. [CORC-4860] DPCPP compiler team argue
// that it violates the spec that kernel name should be
// globally visible if put the kernel name in anonymous namespace.

template <typename Op, typename scalar, typename IndexType, int ADims, int step>
class PointwiseApply1 {};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    int step>
class PointwiseApply2 {};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int step>
class PointwiseApply3 {};

template <
    typename T1,
    typename IndexType,
    typename T2 = void,
    typename T3 = void,
    typename T4 = void>
inline void rearrangeDims(
    detail::TensorInfo<T1, IndexType>* aInfo,
    detail::TensorInfo<T2, IndexType>* bInfo = nullptr,
    detail::TensorInfo<T3, IndexType>* cInfo = nullptr,
    detail::TensorInfo<T4, IndexType>* dInfo = nullptr) {
  int numInfos = 1;
  int dims = aInfo->dims;
  IndexType* sizes[4] = {
      aInfo->sizes,
  };
  IndexType* strides[4] = {
      aInfo->strides,
  };

  if (bInfo != nullptr) {
    ++numInfos;
    if (bInfo->dims != dims)
      return;
    sizes[1] = bInfo->sizes;
    strides[1] = bInfo->strides;
  }

  if (cInfo != nullptr) {
    ++numInfos;
    if (cInfo->dims != dims)
      return;
    sizes[2] = cInfo->sizes;
    strides[2] = cInfo->strides;
  }

  if (dInfo != nullptr) {
    ++numInfos;
    if (dInfo->dims != dims)
      return;
    sizes[3] = dInfo->sizes;
    strides[3] = dInfo->strides;
  }

  // Bail out if sizes do not match: we are using "deprecated pointwise
  // behavior" among tensors of different shapes but same number of elements.
  for (int i = 1; i < numInfos; ++i) {
    for (int j = 0; j < dims; ++j) {
      if (sizes[i][j] != sizes[0][j])
        return;
    }
  }

  for (int i = 0; i < dims - 1; ++i) {
    // No need to consider dimensions of size 1.
    if (sizes[0][i] == 1)
      continue;

    for (int j = i + 1; j < dims; ++j) {
      if (sizes[0][j] == 1)
        continue;

      // Compare the relative sizes of strides between dim #i and dim #j.
      bool hasIncreasingStrides = false;
      bool hasDecreasingStrides = false;

      for (int k = 0; k < numInfos; k++) {
        IndexType stride_i = strides[k][i];
        IndexType stride_j = strides[k][j];
        if (stride_i < stride_j) {
          hasIncreasingStrides = true;
        } else if (stride_i > stride_j) {
          hasDecreasingStrides = true;
        }
      }

      if (hasIncreasingStrides && !hasDecreasingStrides) {
        for (int k = 0; k < numInfos; k++) {
          IndexType size = sizes[k][i];
          sizes[k][i] = sizes[k][j];
          sizes[k][j] = size;

          IndexType stride = strides[k][i];
          strides[k][i] = strides[k][j];
          strides[k][j] = stride;
        }
      }
    }
  }
}

template <
    typename Op,
    typename scalar,
    typename IndexType,
    int ADims,
    bool with_offset,
    int remaining_steps,
    typename... Offsets>
struct ApplyOp1 {
  inline static void apply(
      const detail::TensorInfo<scalar, IndexType>& a,
      const Op& op,
      void *a_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets) {
    // Convert 'linearIndex' into an offset of 'a'
    const IndexType aOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar, IndexType, ADims>::get(linearIndex, a)
        : 0;
    // Convert 'linearIndex' into an offset of input 'b'
    ApplyOp1<
        Op,
        scalar,
        IndexType,
        ADims,
        with_offset,
        remaining_steps - 1,
        const IndexType,
        Offsets...>::
        apply(a, op, a_pointer, n, linearIndex + 1, aOffsets..., aOffset);
  }
};

// Specialize 'step=1' case (i.e., 'remaining_steps = 0' and 'len(Offsets)=1').
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar,
    typename IndexType,
    int ADims,
    typename Offset>
struct ApplyOp1<Op, scalar, IndexType, ADims, false, 0, Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar, IndexType>& a,
      const Op& op,
      void *a_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset) {
    auto a_ptr = (scalar *)a_pointer;
    op(a_ptr[aOffset]);
  }
};

// Specialize 'step=1' case (i.e., 'remaining_steps = 0' and 'len(Offsets)=1').
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar,
    typename IndexType,
    int ADims,
    typename Offset>
struct ApplyOp1<Op, scalar, IndexType, ADims, true, 0, Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar, IndexType>& a,
      const Op& op,
      void *a_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset) {
    auto a_ptr = (scalar *)a_pointer;
    op(a_ptr[aOffset], (int64_t)linearIndex);
  }
};

template <
    typename Op,
    typename scalar,
    typename IndexType,
    int ADims,
    bool with_offset,
    typename... Offsets>
struct ApplyOp1<Op, scalar, IndexType, ADims, with_offset, 0, Offsets...> {
  inline static void apply(
      const detail::TensorInfo<scalar, IndexType>& a,
      const Op& op,
      void *a_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets) {
    auto a_ptr = (scalar *)a_pointer;
    op(n, a_ptr[aOffsets]...);
  }
};

template <
    typename Op,
    typename scalar,
    typename IndexType,
    int ADims,
    int step,
    bool with_offset>
void kernelPointwiseApply1(
    detail::TensorInfo<scalar, IndexType> a,
    IndexType totalElements,
    const Op op) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(totalElements, tileSize, rng, GRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    void *a_pointer = a.data;
    cgh.parallel_for<PointwiseApply1<Op, scalar, IndexType, ADims, step>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(tileSize), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          for (IndexType linearIndex = item.get_global_id(0) * step;
               linearIndex < totalElements;
               linearIndex += item.get_global_range()[0] * step) {
            ApplyOp1<Op, scalar, IndexType, ADims, with_offset, step>::apply(
                a,
                op,
                a_pointer,
                DPCPP::min(step, static_cast<int>(totalElements - linearIndex)),
                linearIndex);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    bool with_offset,
    int remaining_steps,
    typename... Offsets>
struct ApplyOp2 {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets) {
    // Convert 'linearIndex' into an offset of 'a'
    const IndexType aOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a)
        : 0;
    // Convert 'linearIndex' into an offset of input 'b'
    const IndexType bOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b)
        : 0;
    ApplyOp2<
        Op,
        scalar1,
        scalar2,
        IndexType,
        ADims,
        BDims,
        with_offset,
        remaining_steps - 1,
        const IndexType,
        Offsets...>::
        apply(
            a,
            b,
            op,
            a_pointer,
            b_pointer,
            n,
            linearIndex + 1,
            aOffsets...,
            aOffset,
            bOffsets...,
            bOffset);
  }
};

// Specialize 'step=1' case (i.e., 'remaining_steps = 0' and 'len(Offsets)=1').
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    typename Offset>
struct ApplyOp2<
    Op,
    scalar1,
    scalar2,
    IndexType,
    ADims,
    BDims,
    false,
    0,
    Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset,
      Offset bOffset) {
    auto a_ptr = (scalar1 *)a_pointer;
    auto b_ptr = (scalar2 *)b_pointer;
    op(a_ptr[aOffset], b_ptr[bOffset]);
  }
};

// Specialize 'step=1' case (i.e., 'remaining_steps = 0' and 'len(Offsets)=1').
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    typename Offset>
struct ApplyOp2<
    Op,
    scalar1,
    scalar2,
    IndexType,
    ADims,
    BDims,
    true,
    0,
    Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset,
      Offset bOffset) {
    auto a_ptr = (scalar1 *)a_pointer;
    auto b_ptr = (scalar2 *)b_pointer;
    op(a_ptr[aOffset], b_ptr[bOffset], (int64_t)linearIndex);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    bool with_offset,
    typename... Offsets>
struct ApplyOp2<
    Op,
    scalar1,
    scalar2,
    IndexType,
    ADims,
    BDims,
    with_offset,
    0,
    Offsets...> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets) {
    auto a_ptr = (scalar1 *)a_pointer;
    auto b_ptr = (scalar2 *)b_pointer;
    op(n, a_ptr[aOffsets]..., b_ptr[bOffsets]...);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int BDims,
    int step,
    bool with_offset>
void kernelPointwiseApply2(
    detail::TensorInfo<scalar1, IndexType> output,
    detail::TensorInfo<scalar2, IndexType> input,
    IndexType totalElements,
    const Op op) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(totalElements, tileSize, rng, GRange);

  // 1. Initialize temp buffer
  auto cgf = DPCPP_Q_CGF(cgh) {
    void *in_ptr = input.data;
    void *out_ptr = output.data;
    cgh.parallel_for<
        PointwiseApply2<Op, scalar1, scalar2, IndexType, ADims, BDims, step>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(tileSize), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          for (IndexType linearIndex = item.get_global_id(0) * step;
               linearIndex < totalElements;
               linearIndex += item.get_global_range()[0] * step) {
            ApplyOp2<
                Op,
                scalar1,
                scalar2,
                IndexType,
                ADims,
                BDims,
                with_offset,
                step>::
                apply(
                    output,
                    input,
                    op,
                    out_ptr,
                    in_ptr,
                    DPCPP::min(
                        step, static_cast<int>(totalElements - linearIndex)),
                    linearIndex);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int remaining_steps,
    typename... Offsets>
struct ApplyOp3 {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets,
      Offsets... cOffsets) {
    // Convert 'linearIndex' into an offset of 'a'
    const IndexType aOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a)
        : 0;
    // Convert 'linearIndex' into an offset of input 'b'
    const IndexType bOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b)
        : 0;
    // Convert 'linearIndex' into an offset of input 'c'
    const IndexType cOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar3, IndexType, CDims>::get(linearIndex, c)
        : 0;

    ApplyOp3<
        Op,
        scalar1,
        scalar2,
        scalar3,
        IndexType,
        ADims,
        BDims,
        CDims,
        remaining_steps - 1,
        const IndexType,
        Offsets...>::
        apply(
            a,
            b,
            c,
            op,
            a_pointer,
            b_pointer,
            c_pointer,
            n,
            linearIndex + 1,
            aOffsets...,
            aOffset,
            bOffsets...,
            bOffset,
            cOffsets...,
            cOffset);
  }
};

//

// Specialize 'step=1' case (i.e., 'remaining_steps = 0' and 'len(Offsets)=1').
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    typename Offset>
struct ApplyOp3<
    Op,
    scalar1,
    scalar2,
    scalar3,
    IndexType,
    ADims,
    BDims,
    CDims,
    0,
    Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset,
      Offset bOffset,
      Offset cOffset) {
    auto a_ptr = (scalar1*)a_pointer;
    auto b_ptr = (scalar2*)b_pointer;
    auto c_ptr = (scalar3*)c_pointer;
    op(a_ptr[aOffset], b_ptr[bOffset], c_ptr[cOffset]);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    typename... Offsets>
struct ApplyOp3<
    Op,
    scalar1,
    scalar2,
    scalar3,
    IndexType,
    ADims,
    BDims,
    CDims,
    0,
    Offsets...> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets,
      Offsets... cOffsets) {
    auto a_ptr = (scalar1*)a_pointer;
    auto b_ptr = (scalar2*)b_pointer;
    auto c_ptr = (scalar3*)c_pointer;
    op(n, a_ptr[aOffsets]..., b_ptr[bOffsets]..., c_ptr[cOffsets]...);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int step>
void kernelPointwiseApply3(
    detail::TensorInfo<scalar1, IndexType> output,
    detail::TensorInfo<scalar2, IndexType> input1,
    detail::TensorInfo<scalar3, IndexType> input2,
    IndexType totalElements,
    const Op op) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(totalElements, tileSize, rng, GRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    void *in1_ptr = input1.data;
    void *in2_ptr = input2.data;
    void *out_ptr = output.data;
    cgh.parallel_for<PointwiseApply3<
        Op,
        scalar1,
        scalar2,
        scalar3,
        IndexType,
        ADims,
        BDims,
        CDims,
        step>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(tileSize), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          for (IndexType linearIndex = item.get_global_id(0) * step;
               linearIndex < totalElements;
               linearIndex += item.get_global_range()[0] * step) {
            ApplyOp3<
                Op,
                scalar1,
                scalar2,
                scalar3,
                IndexType,
                ADims,
                BDims,
                CDims,
                step>::
                apply(
                    output,
                    input1,
                    input2,
                    op,
                    out_ptr,
                    in1_ptr,
                    in2_ptr,
                    DPCPP::min(
                        step, static_cast<int>(totalElements - linearIndex)),
                    linearIndex);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int DDims,
    int remaining_steps,
    typename... Offsets>
struct ApplyOp4 {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const detail::TensorInfo<scalar3, IndexType>& d,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      void *d_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets,
      Offsets... cOffsets,
      Offsets... dOffsets) {
    // Convert 'linearIndex' into an offset of 'a'
    const IndexType aOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a)
        : 0;
    // Convert 'linearIndex' into an offset of input 'b'
    const IndexType bOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b)
        : 0;
    // Convert 'linearIndex' into an offset of input 'c'
    const IndexType cOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar3, IndexType, CDims>::get(linearIndex, c)
        : 0;
    // Convert 'linearIndex' into an offset of input 'c'
    const IndexType dOffset = static_cast<int>(sizeof...(Offsets)) < n
        ? detail::IndexToOffset<scalar4, IndexType, DDims>::get(linearIndex, d)
        : 0;

    ApplyOp4<
        Op,
        scalar1,
        scalar2,
        scalar3,
        scalar4,
        IndexType,
        ADims,
        BDims,
        CDims,
        DDims,
        remaining_steps - 1,
        const IndexType,
        Offsets...>::
        apply(
            a,
            b,
            c,
            d,
            op,
            a_pointer,
            b_pointer,
            c_pointer,
            d_pointer,
            n,
            linearIndex + 1,
            aOffsets...,
            aOffset,
            bOffsets...,
            bOffset,
            cOffsets...,
            cOffset,
            dOffsets...,
            dOffset);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int DDims,
    typename Offset>
struct ApplyOp4<
    Op,
    scalar1,
    scalar2,
    scalar3,
    scalar4,
    IndexType,
    ADims,
    BDims,
    CDims,
    DDims,
    0,
    Offset> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const detail::TensorInfo<scalar3, IndexType>& d,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      void *d_pointer,
      int n,
      IndexType linearIndex,
      Offset aOffset,
      Offset bOffset,
      Offset cOffset,
      Offset dOffset) {
    auto a_ptr = (scalar1*)a_pointer;
    auto b_ptr = (scalar2*)b_pointer;
    auto c_ptr = (scalar3*)c_pointer;
    auto d_ptr = (scalar4*)d_pointer;
    op(a_ptr[aOffset], b_ptr[bOffset], c_ptr[cOffset], d_ptr[dOffset]);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int DDims,
    typename... Offsets>
struct ApplyOp4<
    Op,
    scalar1,
    scalar2,
    scalar3,
    scalar4,
    IndexType,
    ADims,
    BDims,
    CDims,
    DDims,
    0,
    Offsets...> {
  inline static void apply(
      const detail::TensorInfo<scalar1, IndexType>& a,
      const detail::TensorInfo<scalar2, IndexType>& b,
      const detail::TensorInfo<scalar3, IndexType>& c,
      const detail::TensorInfo<scalar3, IndexType>& d,
      const Op& op,
      void *a_pointer,
      void *b_pointer,
      void *c_pointer,
      void *d_pointer,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets,
      Offsets... cOffsets,
      Offsets... dOffsets) {
    auto a_ptr = (scalar1*)a_pointer;
    auto b_ptr = (scalar2*)b_pointer;
    auto c_ptr = (scalar3*)c_pointer;
    auto d_ptr = (scalar4*)d_pointer;
    op(n,
       a_ptr[aOffsets]...,
       b_ptr[bOffsets]...,
       c_ptr[cOffsets]...,
       d_ptr[dOffsets]...);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int DDims,
    int step>
class PointwiseApply4 {};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename IndexType,
    int ADims,
    int BDims,
    int CDims,
    int DDims,
    int step>
void kernelPointwiseApply4(
    detail::TensorInfo<scalar1, IndexType> output,
    detail::TensorInfo<scalar2, IndexType> input1,
    detail::TensorInfo<scalar3, IndexType> input2,
    detail::TensorInfo<scalar4, IndexType> input3,
    IndexType totalElements,
    const Op op) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(totalElements, tileSize, rng, GRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    void *in1_ptr = input1.data;
    void *in2_ptr = input2.data;
    void *in3_ptr = input3.data;
    void *out_ptr = output.data;
    cgh.parallel_for<PointwiseApply4<
        Op,
        scalar1,
        scalar2,
        scalar3,
        scalar4,
        IndexType,
        ADims,
        BDims,
        CDims,
        DDims,
        step>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(tileSize), DPCPP::range<1>(tileSize)),
        [=](DPCPP::nd_item<1> item) {
          for (IndexType linearIndex = item.get_global_id(0) * step;
               linearIndex < totalElements;
               linearIndex += item.get_global_range()[0] * step) {
            ApplyOp4<
                Op,
                scalar1,
                scalar2,
                scalar3,
                scalar4,
                IndexType,
                ADims,
                BDims,
                CDims,
                DDims,
                step>::
                apply(
                    output,
                    input1,
                    input2,
                    input3,
                    op,
                    out_ptr,
                    in1_ptr,
                    in2_ptr,
                    in3_ptr,
                    DPCPP::min(
                        step, static_cast<int>(totalElements - linearIndex)),
                    linearIndex);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

// } // namespace

template <typename scalar, int step, typename Op, bool with_offset = false>
inline void DPCPP_tensor_apply1(at::Tensor a, const Op& op) {
  checkBackend("DPCPP_Tensor_apply1", {a}, Backend::XPU);
  auto dim = a.dim();

  std::vector<int64_t> collapsed_shape;
  std::vector<int64_t> collapsed_strides;
  collapsed_shape.reserve(dim);
  collapsed_strides.reserve(dim);
  for (int64_t i = 0; i < dim; i++) {
    if (a.stride(i) != 0) {
      collapsed_shape.push_back(a.size(i));
      collapsed_strides.push_back(a.stride(i));
    }
  }

  if (static_cast<int64_t>(collapsed_shape.size()) != dim) {
    a = a.as_strided(collapsed_shape, collapsed_strides);
  }

  int64_t totalElements = a.numel();

  TORCH_CHECK(dim <= MAX_TENSORINFO_DIMS, "dim exceed max allowed dim");

  if (totalElements == 0) {
    return;
  }

  Tensor oldA;
  if (detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }

// It is possible that the tensor dimensions are able to be collapsed,
// and thus we can reduce the actual code complexity of the copy by
// exploiting this knowledge statically, since the div/mod is the
// most expensive part of the operation, more so than memory accesses.
// For instance, when copying a non-contiguous to a contiguous tensor
// (or vice versa), the contiguous tensor can be collapsed to one
// dimension, and the loop to translate the linear index to the array
// index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                     \
  kernelPointwiseApply1<Op, scalar, TYPE, A, step, with_offset>( \
      aInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_A_CASE(TYPE, A) \
  {                            \
    switch (A) {               \
      case 1:                  \
        HANDLE_CASE(TYPE, 1);  \
        break;                 \
      case 2:                  \
        HANDLE_CASE(TYPE, 2);  \
        break;                 \
      default:                 \
        HANDLE_CASE(TYPE, -1); \
        break;                 \
    }                          \
  }

  if (detail::canUse32BitIndexMath(a)) {
    detail::TensorInfo<scalar, unsigned int> aInfo =
        detail::getTensorInfo<scalar, unsigned int>(a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();
    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    detail::TensorInfo<scalar, uint64_t> aInfo =
        detail::getTensorInfo<scalar, uint64_t>(a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();

    if (aInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1);
    } else {
      HANDLE_CASE(uint64_t, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    TORCH_CHECK(0, "not implemented THDPCPPTensor_copyIgnoringOverlaps\n");
    // at::native::legacy::dpcpp::_th_copy_ignoring_overlaps_(oldA, a);
  }
}

/* Provides default step = 1 to DPCPP_tensor_apply1. */
template <typename scalar, typename Op, bool with_offset = false>
inline void DPCPP_tensor_apply1(at::Tensor a, const Op& op) {
  DPCPP_tensor_apply1<scalar, 1, Op, with_offset>(a, op);
}

template <
    typename scalar1,
    typename scalar2,
    int step,
    typename Op,
    bool with_offset = false>
inline void DPCPP_tensor_apply2(at::Tensor dst, at::Tensor src, const Op& op) {
  if(src.is_quantized())
      checkBackend("DPCPP_Tensor_apply2", {dst, src}, Backend::QuantizedXPU);
  else
      checkBackend("DPCPP_Tensor_apply2", {dst, src}, Backend::XPU);
  int64_t totalElements = dst.numel();

  TORCH_CHECK(
      totalElements == src.numel(), "src element is not the same as dst");

  TORCH_CHECK(
      dst.dim() <= MAX_TENSORINFO_DIMS && src.dim() <= MAX_TENSORINFO_DIMS,
      "src or dst dim exceed max tensor dim boundary");

  if (dst.numel() == 0) {
    return;
  }

  Tensor old_dst;
  if (detail::maybeOverlappingIndices(dst)) {
    // Must perform in contiguous space
    old_dst = dst;
    dst = dst.contiguous();
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A, B)                                               \
  kernelPointwiseApply2<Op, scalar1, scalar2, TYPE, A, B, step, with_offset>( \
      dstInfo, srcInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_B_CASE(TYPE, A, B) \
  {                               \
    switch (B) {                  \
      case 1:                     \
        HANDLE_CASE(TYPE, A, 1);  \
        break;                    \
      case 2:                     \
        HANDLE_CASE(TYPE, A, 2);  \
        break;                    \
      default:                    \
        HANDLE_CASE(TYPE, A, -1); \
        break;                    \
    }                             \
  }
#define HANDLE_A_CASE(TYPE, A, B)   \
  {                                 \
    switch (A) {                    \
      case 1:                       \
        HANDLE_B_CASE(TYPE, 1, B);  \
        break;                      \
      case2:                        \
        HANDLE_B_CASE(TYPE, 2, B);  \
        break;                      \
      default:                      \
        HANDLE_B_CASE(TYPE, -1, B); \
        break;                      \
    }                               \
  }

  if (detail::canUse32BitIndexMath(dst) && detail::canUse32BitIndexMath(src)) {
    detail::TensorInfo<scalar1, unsigned int> dstInfo =
        detail::getTensorInfo<scalar1, unsigned int>(dst);

    detail::TensorInfo<scalar2, unsigned int> srcInfo =
        detail::getTensorInfo<scalar2, unsigned int>(src);
    rearrangeDims(&dstInfo, &srcInfo);
    dstInfo.collapseDims();
    srcInfo.collapseDims();
    if (dstInfo.dims == 1 && srcInfo.dims == 1) {
      HANDLE_CASE(unsigned int, 1, 1);
    } else {
      HANDLE_CASE(unsigned int, -1, -1);
    }

    //   HANDLE_A_CASE(unsigned int, dstInfo.dims, srcInfo.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> dstInfo =
        detail::getTensorInfo<scalar1, uint64_t>(dst);
    detail::TensorInfo<scalar2, uint64_t> srcInfo =
        detail::getTensorInfo<scalar2, uint64_t>(src);
    rearrangeDims(&dstInfo, &srcInfo);
    dstInfo.collapseDims();
    srcInfo.collapseDims();

    if (dstInfo.dims == 1 && srcInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (old_dst.defined()) {
    TORCH_CHECK(0, "not implemented THDPCPPTensor_copyIgnoringOverlaps\n");
    // at::native::legacy::dpcpp::_th_copy_ignoring_overlaps_(old_dst, dst);
  }
}

/* Provides default step = 1 to DPCPP_tensor_apply2. */
template <
    typename scalar1,
    typename scalar2,
    typename Op,
    bool with_offset = false>
inline void DPCPP_tensor_apply2(at::Tensor a, at::Tensor b, const Op& op) {
  DPCPP_tensor_apply2<scalar1, scalar2, 1, Op, with_offset>(a, b, op);
}

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    int step,
    typename Op>
inline void DPCPP_tensor_apply3(
    at::Tensor dst,
    at::Tensor src1,
    at::Tensor src2,
    const Op& op) {
  checkBackend("DPCPP_Tensor_apply3", {dst, src1, src2}, Backend::XPU);
  int64_t totalElements = dst.numel();

  TORCH_CHECK(
      totalElements == src1.numel() && totalElements == src2.numel(),
      "src element is not the same as dst");

  TORCH_CHECK(
      dst.dim() <= MAX_TENSORINFO_DIMS && src1.dim() <= MAX_TENSORINFO_DIMS &&
          src2.dim() <= MAX_TENSORINFO_DIMS,
      "src or dst dim exceed max tensor dim boundary");

  if (dst.numel() == 0) {
    return;
  }

  Tensor old_dst;
  if (detail::maybeOverlappingIndices(dst)) {
    // Must perform in contiguous space
    old_dst = dst;
    dst = dst.contiguous();
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A, B, C)                                           \
  kernelPointwiseApply3<Op, scalar1, scalar2, scalar3, TYPE, A, B, C, step>( \
      dstInfo, src1Info, src2Info, static_cast<TYPE>(totalElements), op);

#define HANDLE_C_CASE(TYPE, A, B, C) \
  {                                  \
    switch (C) {                     \
      case 1:                        \
        HANDLE_CASE(TYPE, A, B, 1);  \
        break;                       \
      case 2:                        \
        HANDLE_CASE(TYPE, A, B, 2);  \
        break;                       \
      default:                       \
        HANDLE_CASE(TYPE, A, B, -1); \
        break;                       \
    }                                \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)   \
  {                                    \
    switch (B) {                       \
      case 1:                          \
        HANDLE_C_CASE(TYPE, A, 1, C);  \
        break;                         \
      case 2:                          \
        HANDLE_C_CASE(TYPE, A, 2, C);  \
        break;                         \
      default:                         \
        HANDLE_C_CASE(TYPE, A, -1, C); \
        break;                         \
    }                                  \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)   \
  {                                    \
    switch (A) {                       \
      case 1:                          \
        HANDLE_B_CASE(TYPE, 1, B, C);  \
        break;                         \
      case 2:                          \
        HANDLE_B_CASE(TYPE, 2, B, C);  \
        break;                         \
      default:                         \
        HANDLE_B_CASE(TYPE, -1, B, C); \
        break;                         \
    }                                  \
  }

  if (detail::canUse32BitIndexMath(dst) && detail::canUse32BitIndexMath(src1) &&
      detail::canUse32BitIndexMath(src2)) {
    detail::TensorInfo<scalar1, unsigned int> dstInfo =
        detail::getTensorInfo<scalar1, unsigned int>(dst);

    detail::TensorInfo<scalar2, unsigned int> src1Info =
        detail::getTensorInfo<scalar2, unsigned int>(src1);

    detail::TensorInfo<scalar3, unsigned int> src2Info =
        detail::getTensorInfo<scalar3, unsigned int>(src2);

    rearrangeDims(&dstInfo, &src1Info, &src2Info);
    dstInfo.collapseDims();
    src1Info.collapseDims();
    src2Info.collapseDims();
    if (dstInfo.dims == 1 && src1Info.dims == 1 && src2Info.dims == 1) {
      HANDLE_CASE(unsigned int, 1, 1, 1);
    } else {
      HANDLE_CASE(unsigned int, -1, -1, -1);
    }

    //   HANDLE_A_CASE(unsigned int, dstInfo.dims, src1Info.dims,
    //   src2Info.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> dstInfo =
        detail::getTensorInfo<scalar1, uint64_t>(dst);
    detail::TensorInfo<scalar2, uint64_t> src1Info =
        detail::getTensorInfo<scalar2, uint64_t>(src1);
    detail::TensorInfo<scalar3, uint64_t> src2Info =
        detail::getTensorInfo<scalar3, uint64_t>(src2);

    rearrangeDims(&dstInfo, &src1Info, &src2Info);
    dstInfo.collapseDims();
    src1Info.collapseDims();
    src2Info.collapseDims();

    if (dstInfo.dims == 1 && src1Info.dims == 1 && src2Info.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (old_dst.defined()) {
    TORCH_CHECK(0, "not implemented THDPCPPTensor_copyIgnoringOverlaps\n");
    // at::native::legacy::dpcpp::_th_copy_ignoring_overlaps_(old_dst, dst);
  }
}

template <typename scalar1, typename scalar2, typename scalar3, typename Op>
inline void DPCPP_tensor_apply3(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const Op op) {
  DPCPP_tensor_apply3<scalar1, scalar2, scalar3, 1, Op>(a, b, c, op);
}

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    int step,
    typename Op>
inline void DPCPP_tensor_apply4(
    at::Tensor dst,
    at::Tensor src1,
    at::Tensor src2,
    at::Tensor src3,
    const Op& op) {
  checkBackend("DPCPP_Tensor_apply4", {dst, src1, src2, src3}, Backend::XPU);
  int64_t totalElements = dst.numel();

  TORCH_CHECK(
      totalElements == src1.numel() && totalElements == src2.numel() &&
          totalElements == src3.numel(),
      "src element is not the same as dst");

  TORCH_CHECK(
      dst.dim() <= MAX_TENSORINFO_DIMS && src1.dim() <= MAX_TENSORINFO_DIMS &&
          src2.dim() <= MAX_TENSORINFO_DIMS &&
          src3.dim() <= MAX_TENSORINFO_DIMS,
      "src or dst dim exceed max tensor dim boundary");

  if (dst.numel() == 0) {
    return;
  }

  Tensor old_dst;
  if (detail::maybeOverlappingIndices(dst)) {
    // Must perform in contiguous space
    old_dst = dst;
    dst = dst.contiguous();
  }

#define HANDLE_CASE(TYPE, A, B, C, D)   \
  kernelPointwiseApply4<                \
      Op,                               \
      scalar1,                          \
      scalar2,                          \
      scalar3,                          \
      scalar4,                          \
      TYPE,                             \
      A,                                \
      B,                                \
      C,                                \
      D,                                \
      step>(                            \
      dstInfo,                          \
      src1Info,                         \
      src2Info,                         \
      src3Info,                         \
      static_cast<TYPE>(totalElements), \
      op);

#define HANDLE_D_CASE(TYPE, A, B, C, D) \
  {                                     \
    switch (D) {                        \
      case 1:                           \
        HANDLE_CASE(TYPE, A, B, C, 1);  \
        break;                          \
      case 2:                           \
        HANDLE_CASE(TYPE, A, B, C, 2);  \
        break;                          \
      default:                          \
        HANDLE_CASE(TYPE, A, B, C, -1); \
        break;                          \
    }                                   \
  }

#define HANDLE_C_CASE(TYPE, A, B, C, D)   \
  {                                       \
    switch (C) {                          \
      case 1:                             \
        HANDLE_D_CASE(TYPE, A, B, 1, D);  \
        break;                            \
      case 2:                             \
        HANDLE_D_CASE(TYPE, A, B, 2, D);  \
        break;                            \
      default:                            \
        HANDLE_D_CASE(TYPE, A, B, -1, D); \
        break;                            \
    }                                     \
  }

#define HANDLE_B_CASE(TYPE, A, B, C, D)   \
  {                                       \
    switch (B) {                          \
      case 1:                             \
        HANDLE_C_CASE(TYPE, A, 1, C, D);  \
        break;                            \
      case 2:                             \
        HANDLE_C_CASE(TYPE, A, 2, C, D);  \
        break;                            \
      default:                            \
        HANDLE_C_CASE(TYPE, A, -1, C, D); \
        break;                            \
    }                                     \
  }

#define HANDLE_A_CASE(TYPE, A, B, C, D)   \
  {                                       \
    switch (A) {                          \
      case 1:                             \
        HANDLE_B_CASE(TYPE, 1, B, C, D);  \
        break;                            \
      case 2:                             \
        HANDLE_B_CASE(TYPE, 2, B, C, D);  \
        break;                            \
      default:                            \
        HANDLE_B_CASE(TYPE, -1, B, C, D); \
        break;                            \
    }                                     \
  }

  if (detail::canUse32BitIndexMath(dst) && detail::canUse32BitIndexMath(src1) &&
      detail::canUse32BitIndexMath(src2) &&
      detail::canUse32BitIndexMath(src3)) {
    detail::TensorInfo<scalar1, unsigned int> dstInfo =
        detail::getTensorInfo<scalar1, unsigned int>(dst);

    detail::TensorInfo<scalar2, unsigned int> src1Info =
        detail::getTensorInfo<scalar2, unsigned int>(src1);

    detail::TensorInfo<scalar3, unsigned int> src2Info =
        detail::getTensorInfo<scalar3, unsigned int>(src2);

    detail::TensorInfo<scalar4, unsigned int> src3Info =
        detail::getTensorInfo<scalar4, unsigned int>(src3);

    rearrangeDims(&dstInfo, &src1Info, &src2Info, &src3Info);
    dstInfo.collapseDims();
    src1Info.collapseDims();
    src2Info.collapseDims();
    src3Info.collapseDims();

    if (dstInfo.dims == 1 && src1Info.dims == 1 && src2Info.dims == 1 &&
        src3Info.dims == 1) {
      HANDLE_CASE(unsigned int, 1, 1, 1, 1);
    } else {
      HANDLE_CASE(unsigned int, -1, -1, -1, -1);
    }

    // HANDLE_CASE(uint64_t, dstInfo.dims, src1Info.dims, src2Info.dims,
    // src3Info.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> dstInfo =
        detail::getTensorInfo<scalar1, uint64_t>(dst);

    detail::TensorInfo<scalar2, uint64_t> src1Info =
        detail::getTensorInfo<scalar2, uint64_t>(src1);

    detail::TensorInfo<scalar3, uint64_t> src2Info =
        detail::getTensorInfo<scalar3, uint64_t>(src2);

    detail::TensorInfo<scalar4, uint64_t> src3Info =
        detail::getTensorInfo<scalar4, uint64_t>(src3);

    rearrangeDims(&dstInfo, &src1Info, &src2Info, &src3Info);
    dstInfo.collapseDims();
    src1Info.collapseDims();
    src2Info.collapseDims();
    src3Info.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (dstInfo.dims == 1 && src1Info.dims == 1 && src2Info.dims == 1 &&
        src3Info.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_D_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (old_dst.defined()) {
    TORCH_CHECK(0, "not implemented THDPCPPTensor_copyIgnoringOverlaps\n");
    // at::native::legacy::dpcpp::_th_copy_ignoring_overlaps_(old_dst, dst);
  }
}

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename Op>
inline void DPCPP_tensor_apply4(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& d,
    const Op op) {
  DPCPP_tensor_apply4<scalar1, scalar2, scalar3, scalar4, 1, Op>(
      a, b, c, d, op);
}

} // namespace dpcpp
} // namespace at
