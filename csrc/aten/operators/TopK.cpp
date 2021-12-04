#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/AtenIpexTypeXPU.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "BitonicSort.h"
#include "ScanKernel.h"
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace {

template <typename T>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  using RadixType = uint32_t;
  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2:
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  static inline RadixType convert(float v) {
    RadixType x = *((uint32_t*)&v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
  }

  static inline float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    auto v_de = v ^ mask;
    return *((float*)&v_de);
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(uint8_t v) {
    return v;
  }

  static inline uint8_t deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<int8_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int8_t v) {
    return 128u + v;
  }

  static inline int8_t deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<int16_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int16_t v) {
    return 32768u + v;
  }

  static inline int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int32_t v) {
    return 2147483648u + v;
  }

  static inline int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  using RadixType = uint64_t;

  static inline RadixType convert(int64_t v) {
    // assert(sizeof(int64_t) == 8);
    return 9223372036854775808ull + v;
  }

  static inline int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  using RadixType = uint64_t;

  static inline RadixType convert(double v) {
    RadixType x = *((RadixType*)&v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    auto v_de = v ^ mask;
    return *((double*)&v_de);
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  using RadixType = uint32_t;

  static inline RadixType convert(at::Half v) {
    RadixType x = *((uint16_t*)&v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }

  static inline at::Half deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    auto v_de = v ^ mask;
    return *((at::Half*)&v_de);
  }
};

template <>
struct TopKTypeConfig<at::BFloat16> {
  using RadixType = uint32_t;

  static inline RadixType convert(at::BFloat16 v) {
    RadixType x = *((uint16_t*)&v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }

  static inline at::BFloat16 deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    auto v_de = v ^ mask;
    return *((at::BFloat16*)&v_de);
  }
};

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static inline unsigned int getBitfield(unsigned int val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;
    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static inline unsigned int setBitfield(
      unsigned int val,
      unsigned int toInsert,
      int pos,
      int len) {
    pos &= 0xff;
    len &= 0xff;
    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;
    return (val & ~m) | toInsert;
  }
};

template <>
struct Bitfield<uint64_t> {
  static inline uint64_t getBitfield(uint64_t val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static inline uint64_t setBitfield(
      uint64_t val,
      uint64_t toInsert,
      int pos,
      int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;
    return (val & ~m) | toInsert;
  }
};

// Over what radix we are selecting values
#define RADIX_BITS 2 // digits are base-(2 ^ RADIX_BITS)
#define RADIX_SIZE 4 // 2 ^ RADIX_BITS
#define RADIX_MASK (RADIX_SIZE - 1)

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <typename DataType, typename BitDataType, typename IndexType>
DPCPP_DEVICE void countRadixUsingMask(
    int counts[RADIX_SIZE],
    const dpcpp_local_acc_t<int>& smem_acc,
    BitDataType desired,
    BitDataType desiredMask,
    int radixDigitPos,
    IndexType sliceSize,
    IndexType withinSliceStride,
    const dpcpp_global_ptr_pt<DataType>& data_ptr,
    const DPCPP::nd_item<1>& item_id) {
  // Clear out per-thread counts from a previous round
  auto local_id = item_id.get_local_id(0);
  for (int i = 0; i < RADIX_SIZE; ++i) {
    counts[i] = 0;
  }
  if (local_id == 0) {
    for (unsigned int i = 0; i < RADIX_SIZE; i++) {
      smem_acc[i] = 0;
    }
  }

  item_id.barrier(dpcpp_local_fence);

  // scan over all the data, counts per each digit, maybe optimized
  // with dpcpp subgroup in the future.
  for (IndexType i = local_id; i < sliceSize; i += item_id.get_local_range(0)) {
    BitDataType val =
        TopKTypeConfig<DataType>::convert(data_ptr[i * withinSliceStride]);
    BitDataType digitInRadix =
        Bitfield<BitDataType>::getBitfield(val, radixDigitPos, RADIX_BITS);

    if ((val & desiredMask) == desired)
      counts[digitInRadix]++;
  }

  for (unsigned int i = 0; i < RADIX_SIZE; i++) {
    DPCPP::atomic<int, dpcpp_local_space> smem_var(smem_acc.get_pointer() + i);
    smem_var.fetch_add(counts[i]);
  }

  item_id.barrier(dpcpp_local_fence);

  // For each thread, read in the total counts
  for (unsigned int i = 0; i < RADIX_SIZE; ++i) {
    counts[i] = smem_acc[i];
  }
  item_id.barrier(dpcpp_local_fence);
}

// This finds the unique value 'v' that matches the pattern
// ((v & desired) == desiredMask) in our sorted in format
template <typename DataType, typename BitDataType, typename IndexType>
DPCPP_DEVICE DataType findPattern(
    const dpcpp_local_acc_t<int>& smem_acc,
    const dpcpp_global_ptr_pt<DataType>& data_ptr,
    IndexType sliceSize,
    IndexType withinSliceStride,
    BitDataType desired,
    BitDataType desiredMask,
    const DPCPP::nd_item<1>& item_id) {
  auto local_id = item_id.get_local_id(0);
  auto smem_ptr =
      static_cast<DataType*>(static_cast<void*>(smem_acc.get_pointer().get()));
  if (local_id < RADIX_SIZE) {
    smem_ptr[RADIX_SIZE] = ScalarConvert<int, DataType>::to(0);
  }
  item_id.barrier(dpcpp_local_fence);
  // All threads participate in the loop, in order to sync on the flag
  IndexType numIterations =
      RoundUp(sliceSize, (IndexType)item_id.get_local_range(0));
  for (IndexType i = local_id; i < numIterations;
       i += item_id.get_local_range(0)) {
    bool inRange = (i < sliceSize);
    DataType v = inRange ? data_ptr[i * withinSliceStride]
                         : ScalarConvert<int, DataType>::to(0);
    if (inRange &&
        ((TopKTypeConfig<DataType>::convert(v) & desiredMask) == desired)) {
      smem_ptr[0] = ScalarConvert<int, DataType>::to(1);
      smem_ptr[1] = v;
    }
  }

  item_id.barrier(dpcpp_local_fence);

  auto found = smem_ptr[0];
  auto val = smem_ptr[1];

  item_id.barrier(dpcpp_local_fence);
  if (Numerics<DataType>::ne(found, ScalarConvert<int, DataType>::to(0))) {
    return val;
  }

  return ScalarConvert<int, DataType>::to(0);
}

template <
    typename DataType,
    typename BitDataType,
    typename IndexType,
    bool Order>
DPCPP_DEVICE void radixSelect(
    const dpcpp_global_ptr_pt<DataType>& data_ptr,
    const IndexType k,
    const IndexType sliceSize,
    const IndexType withinSliceStride,
    const dpcpp_local_acc_t<int>& smem_acc,
    DataType* topK,
    const DPCPP::nd_item<1>& item_id) {
  // Per-thread buckets into which we accumulate digit counts in our radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  BitDataType desired = 0;
  BitDataType desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(DataType) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<DataType, BitDataType, IndexType>(
        counts,
        smem_acc,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data_ptr,
        item_id);
// All threads participate in the comparisions below to know the
// final result
#define CHECK_RADIX(i)                                                        \
  int count = counts[i];                                                      \
                                                                              \
  /* All threads have the same value in counts here, so all */                \
  /* threads will return from the function. */                                \
  if (count == 1 && kToFind == 1) {                                           \
    /* There is a unique answer. */                                           \
    desired =                                                                 \
        Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS); \
    desiredMask = Bitfield<BitDataType>::setBitfield(                         \
        desiredMask, RADIX_MASK, digitPos, RADIX_BITS);                       \
    /* The answer is now the unique element v such that: */                   \
    /* (v & desiredMask) == desired */                                        \
    /* However, we do not yet know what the actual element is. We */          \
    /* need to perform a search through the data to find the */               \
    /* element that matches this pattern. */                                  \
    *topK = findPattern<DataType, BitDataType, IndexType>(                    \
        smem_acc,                                                             \
        data_ptr,                                                             \
        sliceSize,                                                            \
        withinSliceStride,                                                    \
        desired,                                                              \
        desiredMask,                                                          \
        item_id);                                                             \
    return;                                                                   \
  }                                                                           \
                                                                              \
  if (count >= kToFind) {                                                     \
    desired =                                                                 \
        Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS); \
    desiredMask = Bitfield<BitDataType>::setBitfield(                         \
        desiredMask, RADIX_MASK, digitPos, RADIX_BITS);                       \
                                                                              \
    /* The top-Kth element v must now be one such that: */                    \
    /* (v & desiredMask == desired) */                                        \
    /* but we haven't narrowed it down; we must check the next */             \
    /* least-significant digit */                                             \
    break;                                                                    \
  }                                                                           \
                                                                              \
  kToFind -= count;

    if (Order) {
      // Process in descending order
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        CHECK_RADIX(i);
      }
    } else {
      // Process in ascending order
      for (int i = 0; i < RADIX_SIZE; ++i) {
        CHECK_RADIX(i);
      }
    }
#undef CHECK_RADIX
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = TopKTypeConfig<DataType>::deconvert(desired);
}

template <typename T, typename IndexType, int Dim, bool Order>
void gatherTopK(
    TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`
    IndexType numInputSlices,
    IndexType inputWithinSliceStride,
    TensorInfo<T, IndexType> topK,
    IndexType numTopKSlices,
    IndexType topKWithinSliceStride,
    TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = input.data;
    auto topk_data = topK.data;
    auto indices_data = indices.data;
    auto smem_acc = dpcpp_local_acc_t<int>(32, cgh);
    auto smem_scan_acc = dpcpp_local_acc_t<int>(local_size, cgh);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      IndexType local_id = item_id.get_local_id(0);
      // Find the start offset for our slice
      IndexType slice = item_id.get_group_linear_id();
      IndexType sliceStartIndex =
          IndexToOffset<T, IndexType, Dim>::get(slice, input);
      IndexType topKSliceStartIndex =
          IndexToOffset<T, IndexType, Dim>::get(slice, topK);
      IndexType indicesSliceStartIndex =
          IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);
      T* inputSliceStart = in_data + sliceStartIndex;
      T* topKSliceStart = topk_data + topKSliceStartIndex;
      int64_t* indicesSliceStart = indices_data + indicesSliceStartIndex;
      // Find the k-th highest element in our input
      T topKValue = 0;
      radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
          (dpcpp_global_ptr_pt<T>)inputSliceStart,
          outputSliceSize,
          inputSliceSize,
          inputWithinSliceStride,
          smem_acc,
          &topKValue,
          item_id);
      // (depending on sort dir) in sorted int format is in the top-K.
      // The top-K value itself might not be unique.
      //
      // Since there are a variable number of elements that we see that
      // are within the top-k, we don't know at what index to write out
      // the resulting values.
      // In order to get this, we perform an exclusive prefix sum of
      // `hasTopK`. This will return the resulting index into which we
      // need to write the result, if a thread has a result.
      IndexType numIterations =
          RoundUp(inputSliceSize, (IndexType)item_id.get_local_range(0));
      IndexType writeIndexStart = 0;
      for (IndexType i = local_id; i < numIterations;
           i += item_id.get_local_range(0)) {
        bool inRange = (i < inputSliceSize);
        T v = inRange ? inputSliceStart[i * inputWithinSliceStride]
                      : ScalarConvert<int, T>::to(0);
        bool hasTopK;
        if (Order) {
          hasTopK = inRange && (Numerics<T>::gt(v, topKValue));
        } else {
          hasTopK = inRange && (Numerics<T>::lt(v, topKValue));
        }
        int index;
        int carry;
        exclusivePrefixScan<int>(
            smem_scan_acc, hasTopK, &index, &carry, AddOp<int>(), item_id);
        if (hasTopK) {
          int writeIndex = writeIndexStart + index;

          IndexType topKOffset = writeIndex * topKWithinSliceStride;
          IndexType indexOffset = writeIndex * indicesWithinSliceStride;

          topKSliceStart[topKOffset] = v;
          indicesSliceStart[indexOffset] = i; // to Lua index
        }

        writeIndexStart += carry;
      }

      // We need to fill in the rest with actual == top-K values.
      // The number that we need is outputSliceSize -
      // writeIndexStart. There might be more than that number available,
      // in which case we have to choose the first seen set. We do this
      // via a prefix sum to calculate indices for writing results.
      IndexType topKRemaining = (outputSliceSize - writeIndexStart);
      for (IndexType i = local_id; i < numIterations;
           i += item_id.get_local_range(0)) {
        bool inRange = (i < inputSliceSize);
        T v = inRange ? inputSliceStart[i * inputWithinSliceStride]
                      : ScalarConvert<int, T>::to(0);
        bool hasTopK = inRange && Numerics<T>::eq(v, topKValue);
        int index;
        int carry;
        exclusivePrefixScan<int>(
            smem_scan_acc, hasTopK, &index, &carry, AddOp<int>(), item_id);

        if (hasTopK && static_cast<IndexType>(index) < topKRemaining) {
          int writeIndex = writeIndexStart + index;

          IndexType topKOffset = writeIndex * topKWithinSliceStride;
          IndexType indexOffset = writeIndex * indicesWithinSliceStride;

          topKSliceStart[topKOffset] = v;
          indicesSliceStart[indexOffset] = i;
        }

        if (static_cast<IndexType>(carry) >= topKRemaining) {
          break;
        }

        topKRemaining -= carry;
        writeIndexStart += carry;
      }
    };

    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(numInputSlices * local_size),
            DPCPP::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void Topk(
    Tensor& topK,
    Tensor& indices,
    const Tensor& input_,
    int64_t k,
    int dim,
    int dir,
    int sorted) {
  TORCH_CHECK(
      topK.defined() && indices.defined() && input_.defined(),
      "invalid inputs");
  TORCH_CHECK(topK.dim() <= MAX_DPCPPTORCH_DIMS, "invalid topK dim");
  int64_t dims = indices.dim() == 0 ? 1 : indices.dim();
  TORCH_CHECK(dims <= MAX_DPCPPTORCH_DIMS, "invalid indices dim");
  int numDims = input_.dim() == 0 ? 1 : input_.dim();
  TORCH_CHECK(numDims <= MAX_DPCPPTORCH_DIMS, "invalid input dim");

  TORCH_CHECK(dim >= 0 && dim < numDims, "dim not in range");

  int64_t sliceSize = input_.dim() == 0 ? 1 : input_.size(dim);
  TORCH_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");

  auto input = input_.contiguous();

  // Build the output size, which is the dim being selected set to
  // size k
  std::vector<int64_t> topKSize = {1};
  if (input.dim() != 0)
    topKSize = input.sizes().vec();
  topKSize[dim] = k;
  at::AtenIpexTypeXPU::resize_(topK, topKSize, c10::nullopt);
  at::AtenIpexTypeXPU::resize_(indices, topKSize, c10::nullopt);

// static_cast is required to ensure that the correct type (INDEX_T)
// is provided to the kernel for the arguments.
#define RUN_K(INDEX_T, DIM, DIR)                                 \
  gatherTopK<scalar_t, INDEX_T, DIM, DIR>(                       \
      inputInfo,                                                 \
      static_cast<INDEX_T>(sliceSize),                           \
      static_cast<INDEX_T>(k),                                   \
      static_cast<INDEX_T>(inputSlices),                         \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]), \
      topKInfo,                                                  \
      static_cast<INDEX_T>(topKSlices),                          \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),   \
      indicesInfo,                                               \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]))

#define RUN_DIR(INDEX_T, DIM)   \
  if (dir) {                    \
    RUN_K(INDEX_T, DIM, true);  \
  } else {                      \
    RUN_K(INDEX_T, DIM, false); \
  }

#define RUN_DIM(INDEX_T)     \
  if (allDims == 1) {        \
    RUN_DIR(INDEX_T, 1);     \
  } else if (allDims == 2) { \
    RUN_DIR(INDEX_T, 2);     \
  } else if (allDims == 3) { \
    RUN_DIR(INDEX_T, 3);     \
  } else {                   \
    RUN_DIR(INDEX_T, -1);    \
  }

#define RUN_T(INDEX_T)                                             \
  TensorInfo<scalar_t, INDEX_T> inputInfo =                        \
      getTensorInfo<scalar_t, INDEX_T>(input);                     \
  TensorInfo<scalar_t, INDEX_T> topKInfo =                         \
      getTensorInfo<scalar_t, INDEX_T>(topK);                      \
  TensorInfo<int64_t, INDEX_T> indicesInfo =                       \
      getTensorInfo<int64_t, INDEX_T>(indices);                    \
                                                                   \
  /* We use these structures solely to find the offset to */       \
  /* each slice we are operating on */                             \
  inputInfo.sizes[dim] = 1;                                        \
  topKInfo.sizes[dim] = 1;                                         \
  indicesInfo.sizes[dim] = 1;                                      \
                                                                   \
  /* Collapse all other dims */                                    \
  int collapseInputDim = inputInfo.collapseDims(dim);              \
  int collapseTopKDim = topKInfo.collapseDims(dim);                \
  int collapseIndicesDim = indicesInfo.collapseDims(dim);          \
                                                                   \
  int64_t inputSlices = 1;                                         \
  for (int i = 0; i < inputInfo.dims; ++i) {                       \
    inputSlices *= inputInfo.sizes[i];                             \
  }                                                                \
  int64_t topKSlices = 1;                                          \
  for (int i = 0; i < topKInfo.dims; ++i) {                        \
    topKSlices *= topKInfo.sizes[i];                               \
  }                                                                \
                                                                   \
  /* This is used as a template parameter to calculate indices. */ \
  /* We only specialize it if all collapsed dim sizes are the */   \
  /* same; otherwise, we use -1 which is the specialization */     \
  /* parameter for arbitrary dimensions */                         \
  int allDims = inputInfo.dims;                                    \
  if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {   \
    allDims = -1;                                                  \
  }                                                                \
                                                                   \
  RUN_DIM(INDEX_T);

  if (input.numel() > 0) {
    // Based on required index size, run the algorithm with the
    // appropriate index type
    if (canUse32BitIndexMath(input) && canUse32BitIndexMath(topK) &&
        canUse32BitIndexMath(indices)) {
      RUN_T(uint32_t);
    } else {
      RUN_T(uint64_t);
    }
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_DIR
#undef RUN_K

  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted) {
    if (k <= 2048) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice
      SortKeyValueInplace<scalar_t>(topK, indices, dim, dir);
    } else {
      TORCH_CHECK(
          false,
          "DPCPP can not support element number to sort is larger than 2048");
    }
  }
}

} // namespace

std::tuple<at::Tensor&, at::Tensor&> topk_out(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto dim_ = maybe_wrap_dim(dim, TensorImpl_Unwrap(self));
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "Topk",
      [&]() {
        Topk<scalar_t>(values, indices, self, k, dim_, largest, sorted);
      });

  return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  return at::native::topk(self, k, dim, largest, sorted);
}

} // namespace AtenIpexTypeXPU
} // namespace at
