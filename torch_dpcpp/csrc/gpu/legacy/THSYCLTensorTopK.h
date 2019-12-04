#ifndef THSYCL_TENSOR_TOPK_H
#define THSYCL_TENSOR_TOPK_H
#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>


namespace {
static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
static constexpr auto read_mode = cl::sycl::access::mode::read;
template <typename T>
using local_accessor_t = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

template <typename T>
using sycl_pointer_t = typename cl::sycl::global_ptr<T>::pointer_t;
 

template <typename T>
struct TopKTypeConfig {};

template<>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;
  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2: 
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  static inline RadixType convert(float v)  {
    RadixType x = *((uint32_t*)&v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
  }

  static inline  float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    auto v_de = v ^ mask;
    return *((float*)&v_de);
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  typedef uint32_t RadixType;

  static inline  RadixType convert(uint8_t v) {
    return v;
  }

  static inline  uint8_t deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<int8_t> {
  typedef uint32_t RadixType;

  static inline  RadixType convert(int8_t v) {
    return 128u + v;
  }

  static inline  int8_t deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<int16_t> {
  typedef uint32_t RadixType;

  static inline  RadixType convert(int16_t v) {
    return 32768u + v;
  }

  static inline int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline  RadixType convert(int32_t v) {
    return 2147483648u + v;
  }

  static inline int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline RadixType convert(int64_t v) {
    //assert(sizeof(int64_t) == 8);
    return 9223372036854775808ull + v;
  }

  static inline int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline RadixType convert(double v) {
    RadixType x =  *((uint64_t*)&v);
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
  typedef uint32_t RadixType;

  static inline  RadixType convert(at::Half v) {
    RadixType x =  *((uint16_t*)&v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }

  static inline at::Half deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    auto v_de = v ^ mask;
    return *((at::Half*)&v_de);
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

  static inline unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
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

  static inline uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;
    
    uint64_t m = (1u << len) - 1u;
    toInsert &=m;
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
template <typename DataType, typename BitDataType,
      typename IndexType>
void countRadixUsingMask(int counts[RADIX_SIZE],
             const local_accessor_t<int> &smem_acc,
             BitDataType desired,
             BitDataType desiredMask,
             int radixDigitPos,
             IndexType sliceSize,
             IndexType withinSliceStride,
             const sycl_pointer_t<DataType> &data_ptr,
             const cl::sycl::nd_item<1> &item_id) {
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


  item_id.barrier(cl::sycl::access::fence_space::local_space);
  

  // scan over all the data, counts per each digit, maybe optimized
  // with sycl subgroup in the future.
  for (IndexType i = local_id; i < sliceSize; i += item_id.get_local_range(0)) {
    BitDataType val = TopKTypeConfig<DataType>::convert(data_ptr[i * withinSliceStride]);
    BitDataType digitInRadix = Bitfield<BitDataType>::getBitfield(val, radixDigitPos, RADIX_BITS);
    
    if ((val & desiredMask) == desired)
    counts[digitInRadix]++;
  }

  for (unsigned int i = 0; i < RADIX_SIZE; i++) {
    cl::sycl::atomic<int, cl::sycl::access::address_space::local_space> smem_var(smem_acc.get_pointer() + i);
    smem_var.fetch_add(counts[i]);
  }

  item_id.barrier(cl::sycl::access::fence_space::local_space);

  // For each thread, read in the total counts
  for (unsigned int i = 0; i < RADIX_SIZE; ++i) {
    counts[i] = smem_acc[i];
  }
  item_id.barrier(cl::sycl::access::fence_space::local_space);
}



// This finds the unique value 'v' that matches the pattern
// ((v & desired) == desiredMask) in our sorted in format
template <typename DataType, typename BitDataType, typename IndexType>
DataType findPattern(const local_accessor_t<int> &smem_acc,
           const sycl_pointer_t<DataType> &data_ptr,
           IndexType sliceSize,
           IndexType withinSliceStride,
           BitDataType desired,
           BitDataType desiredMask,
           const cl::sycl::nd_item<1> &item_id) {

  auto local_id = item_id.get_local_id(0);
  auto smem_ptr = SyclConvertToActualTypePtr(DataType, smem_acc);
  if (local_id < RADIX_SIZE) {
    smem_ptr[RADIX_SIZE] = ScalarConvert<int, DataType>::to(0);
  }
  item_id.barrier(cl::sycl::access::fence_space::local_space);
  // All threads participate in the loop, in order to sync on the flag
  IndexType numIterations = THSYCLRoundUp(sliceSize, (IndexType) item_id.get_local_range(0));
  for (IndexType i = local_id; i < numIterations; i += item_id.get_local_range(0)) {
    bool inRange = (i < sliceSize);
  DataType v = inRange ? data_ptr[i*withinSliceStride] : ScalarConvert<int, DataType>::to(0);
    if (inRange && ((TopKTypeConfig<DataType>::convert(v) & desiredMask) == desired)) {
     smem_ptr[0] = ScalarConvert<int, DataType>::to(1);
     smem_ptr[1] = v;
  }
  }

  item_id.barrier(cl::sycl::access::fence_space::local_space); 

  auto found = smem_ptr[0];
  auto val = smem_ptr[1];
  
  item_id.barrier(cl::sycl::access::fence_space::local_space);
  if (THSYCLNumerics<DataType>::ne(found, ScalarConvert<int, DataType>::to(0))) {
    return val;
  }
  
  return ScalarConvert<int, DataType>::to(0);
}



template <typename DataType, typename BitDataType, typename IndexType, bool Order>
void radixSelect(const sycl_pointer_t<DataType> &data_ptr, 
                 const IndexType k,
                 const IndexType sliceSize,
                 const IndexType withinSliceStride,
                 const local_accessor_t<int> &smem_acc,
                 DataType* topK,
                 const cl::sycl::nd_item<1> &item_id) {
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
  for (int digitPos = sizeof(DataType) * 8 - RADIX_BITS;
     digitPos >= 0;
     digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
  // across all threads
  countRadixUsingMask<DataType, BitDataType,IndexType> (
    counts, smem_acc, desired, desiredMask, digitPos,
    sliceSize, withinSliceStride, data_ptr, item_id); 
    // All threads participate in the comparisions below to know the
    // final result
#define CHECK_RADIX(i)                                                  \
    int count = counts[i];                                              \
                                                                        \
    /* All threads have the same value in counts here, so all */        \
    /* threads will return from the function. */                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer. */                                   \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
      /* The answer is now the unique element v such that: */           \
      /* (v & desiredMask) == desired */                                \
      /* However, we do not yet know what the actual element is. We */  \
      /* need to perform a search through the data to find the */       \
      /* element that matches this pattern. */                          \
      *topK = findPattern<DataType, BitDataType, IndexType>(            \
        smem_acc, data_ptr, sliceSize,                                  \
        withinSliceStride, desired, desiredMask, item_id);              \
      return;                                                           \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The top-Kth element v must now be one such that: */            \
      /* (v & desiredMask == desired) */                                \
      /* but we haven't narrowed it down; we must check the next */     \
      /* least-significant digit */                                     \
      break;                                                            \
    }                                                                   \
                                                                        \
    kToFind -= count;                                                   \

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
class gatherTopKKernelName {};

template <typename T, typename IndexType, int Dim, bool Order>
void gatherTopK(TensorInfo<T, IndexType> input, 
                            IndexType inputSliceSize,
                            IndexType outputSliceSize, // aka `k` 
 
                            IndexType numInputSlices,
                            IndexType inputWithinSliceStride,
 
                            TensorInfo<T, IndexType> topK,
                            IndexType numTopKSlices,
                            IndexType topKWithinSliceStride,
 
                            TensorInfo<int64_t, IndexType> indices,
                            IndexType indicesWithinSliceStride) {
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t local_size = sycl_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>();
  sycl_queue.submit([&](cl::sycl::handler &cgh) {
    auto in_acc = c10::sycl::SYCLAccessor<read_mode>(cgh, input.data);
    auto topk_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, topK.data);
    auto indices_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, indices.data);
  auto smem_acc = local_accessor_t<int>(32, cgh);
    auto smem_scan_acc = local_accessor_t<int>(local_size, cgh);
    cgh.parallel_for<gatherTopKKernelName<T, IndexType, Dim, Order> > (
        cl::sycl::nd_range<1>(cl::sycl::range<1>(numInputSlices * local_size), cl::sycl::range<1>(local_size)),
        [=](cl::sycl::nd_item<1> item_id) {
          IndexType local_id = item_id.get_local_id(0);
          // Find the start offset for our slice  
      IndexType slice = item_id.get_group_linear_id();
          IndexType sliceStartIndex =
            IndexToOffset<T, IndexType, Dim>::get(slice, input);
          IndexType topKSliceStartIndex =
            IndexToOffset<T, IndexType, Dim>::get(slice, topK);
          IndexType indicesSliceStartIndex =
            IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);
          T* inputSliceStart = in_acc.template get_pointer<T>() + sliceStartIndex ;
          T* topKSliceStart = topk_acc.template get_pointer<T>() + topKSliceStartIndex;
      int64_t* indicesSliceStart = indices_acc.template get_pointer<int64_t>() + indicesSliceStartIndex;
          // Find the k-th highest element in our input
          T topKValue = 0;
          radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>((sycl_pointer_t<T>)inputSliceStart, outputSliceSize, inputSliceSize, inputWithinSliceStride, smem_acc, &topKValue, item_id);
          // (depending on sort dir) in sorted int format is in the top-K.
          // The top-K value itself might not be unique.
          //  
          // Since there are a variable number of elements that we see that
          // are within the top-k, we don't know at what index to write out 
          // the resulting values.
          // In order to get this, we perform an exclusive prefix sum of
          // `hasTopK`. This will return the resulting index into which we
          // need to write the result, if a thread has a result.
          IndexType numIterations = THSYCLRoundUp(inputSliceSize, (IndexType) item_id.get_local_range(0));
      IndexType writeIndexStart = 0;
        for (IndexType i = local_id; i < numIterations; i += item_id.get_local_range(0)) {
            bool inRange = (i < inputSliceSize);
            T v = 
              inRange ? inputSliceStart[i * inputWithinSliceStride]: ScalarConvert<int, T>::to(0);
            bool hasTopK;
            if (Order) {
              hasTopK = inRange && (THSYCLNumerics<T>::gt(v, topKValue));
            } else {
              hasTopK = inRange && (THSYCLNumerics<T>::lt(v, topKValue));
            }
            int index;
            int carry;
            exclusivePrefixScan<int>(smem_scan_acc, hasTopK, &index, &carry, AddOp<int>(), item_id);
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
         for (IndexType i = local_id; i < numIterations; i += item_id.get_local_range(0)) {
           bool inRange = (i < inputSliceSize);
           T v =
             inRange ? inputSliceStart[i * inputWithinSliceStride] : ScalarConvert<int, T>::to(0);
       bool hasTopK = inRange && THSYCLNumerics<T>::eq(v, topKValue);
           int index;
           int carry;
           exclusivePrefixScan<int>(smem_scan_acc, hasTopK, &index, &carry, AddOp<int>(), item_id);

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

        });
  });


}

} //namepsace
#endif
