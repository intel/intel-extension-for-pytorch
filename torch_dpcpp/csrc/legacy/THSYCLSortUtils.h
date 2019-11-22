#ifndef THSYCL_SORT_UTILS_INC
#define THSYCL_SORT_UTILS_INC
#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>
#include <THDP/THSYCLTensorTypeUtils.h>
#include <THDP/THSYCLNumerics.h>

static const auto write_mode = cl::sycl::access::mode::discard_write;
static const auto read_mode = cl::sycl::access::mode::read;
static const auto read_write_mode = cl::sycl::access::mode::read_write;

template <typename T>
using local_accessor_t = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

// Collection of kernel sort routimes
// Collection of kernel sort routines
template <typename T>
struct LTComp {
  inline bool operator()(const T& a, const T& b) const {
    return THSYCLNumerics<T>::lt(a, b);
  }
};

template <typename T>
struct GTComp {
  inline bool operator()(const T& a, const T& b) const {
    return THSYCLNumerics<T>::gt(a, b);
  }
};

template <typename T>
inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}


template <typename Comparator, typename K, typename V>
inline void bitonicSwap(K& kA, V& vA, bool& validA,
                        K& kB, V& vB, bool& validB,
                        bool dir, const Comparator& comp) {
  // Invalid entries always sort to the end 
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};


template <typename Comparator, typename K, typename V,
         typename IndexType, int Power2SortSize>
inline void bitonicSort(const local_accessor_t<K> &keys_smem,
                        const local_accessor_t<V> &values_smem,
                        const local_accessor_t<bool> &valid_smem,
                        const Comparator& comp,
                        const cl::sycl::nd_item<1> &item_id) {
  auto thread_id = item_id.get_local_id(0);
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((thread_id & (size / 2)) != 0);
    
    for (unsigned int stride = size / 2; stride > 0; stride /=2 ) {
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      
      unsigned int pos  = 2 * thread_id - (thread_id & (stride -1));
      bitonicSwap<Comparator, K, V>(keys_smem[pos], values_smem[pos], valid_smem[pos],
      keys_smem[pos + stride], values_smem[pos + stride], valid_smem[pos + stride], flag, comp);

    }
  }

  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    item_id.barrier(cl::sycl::access::fence_space::local_space);
   
    unsigned int pos = 2 * thread_id - (thread_id & (stride - 1));
    bitonicSwap<Comparator, K, V>(keys_smem[pos], values_smem[pos], valid_smem[pos],
      keys_smem[pos + stride], values_smem[pos + stride], valid_smem[pos + stride], false, comp);
  }

  item_id.barrier(cl::sycl::access::fence_space::local_space);
}


template <typename K, typename V,
          int KeyDims, int ValueDims,
          typename Comparator, typename IndexType, int Power2SortSize>
class binarySortKVInplaceKernelName {};

template <typename K, typename V,
          int KeyDims, int ValueDims,
          typename Comparator, typename IndexType, int Power2SortSize>
void bitonicSortKVInPlace(TensorInfo<K, IndexType> keys,
                          IndexType keySlices,
                          IndexType keySliceSize,
                          IndexType keySliceStride,
                          TensorInfo<V, IndexType> values,
                          IndexType valueSliceStride,
                          Comparator comp) {
  // Find the slice of the tensor that we are sorting
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t local_size = Power2SortSize / 2;
  int64_t global_size = keySlices * local_size;
  sycl_queue.submit([&](cl::sycl::handler &cgh) {
    auto keys_acc = c10::sycl::SYCLAccessor<read_write_mode>(cgh, keys.data);
    auto values_acc = c10::sycl::SYCLAccessor<read_write_mode>(cgh, values.data);
    auto sharedKeys_acc = local_accessor_t<K>(Power2SortSize, cgh);
    auto sharedValues_acc = local_accessor_t<V>(Power2SortSize, cgh);
    auto sharedValid_acc = local_accessor_t<bool>(Power2SortSize, cgh);
    
    cgh.parallel_for<binarySortKVInplaceKernelName<K, V, KeyDims, ValueDims, Comparator, IndexType, Power2SortSize>> (
        cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)), 
        [=](cl::sycl::nd_item<1> item_id){
           auto thread_id = item_id.get_local_id(0);
           auto group_id = item_id.get_group_linear_id();
           const IndexType keyStartOffset = IndexToOffset<K, IndexType, KeyDims>::get(group_id, keys);
           const IndexType valueStartOffset =IndexToOffset<V, IndexType, ValueDims>::get(group_id, values);
           auto keys_ptr = keys_acc.template get_pointer<K>() + keyStartOffset;
           auto values_ptr = values_acc.template get_pointer<V>() + valueStartOffset;
          // If the sort size is 1, the data is already sorted
           if (Power2SortSize == 1) {
             return;
           } else {
             // Otherwise, each thread is responsible for loading and storing 2 
             // elements. The sort size is guaranteed to be >= 2 
             const int elem1 = thread_id;
             const int elem2 = thread_id + (Power2SortSize / 2);

             bool valid1 = (static_cast<IndexType>(elem1) < keySliceSize);
             K k1 = valid1 ? keys_ptr[elem1 * keySliceStride] : ScalarConvert<int, K>::to(0);
             V v1 =  valid1 ? values_ptr[elem1 * valueSliceStride] : ScalarConvert<int, V>::to(0);
             sharedKeys_acc[elem1] = k1;
             sharedValues_acc[elem1] = v1;
             sharedValid_acc[elem1] = valid1;
     
             
             bool valid2 = (static_cast<IndexType>(elem2) < keySliceSize);
             K k2 = valid2 ? keys_ptr[elem2 * keySliceStride] : ScalarConvert<int, K>::to(0);
             V v2 = valid2 ? values_ptr[elem2 * valueSliceStride] : ScalarConvert<int, V>::to(0);
             sharedKeys_acc[elem2] = k2;
             sharedValues_acc[elem2] = v2;
             sharedValid_acc[elem2] = valid2;
             // Sort!
             bitonicSort<Comparator, K, V, IndexType, Power2SortSize>(sharedKeys_acc, sharedValues_acc, sharedValid_acc, comp, item_id);
 
             // elem1 and elem2 values might be out-of-range, if the data size we are sorting is smaller than half the power2 size
             if (valid1) {
               keys_ptr[elem1 * keySliceStride] = sharedKeys_acc[elem1];
               values_ptr[elem1 * valueSliceStride] = sharedValues_acc[elem1];
             }
             if (valid2) {
               keys_ptr[elem2 * keySliceStride] = sharedKeys_acc[elem2];
               values_ptr[elem2 * valueSliceStride] = sharedValues_acc[elem2];
             }
 
           }
    });

  });
  
}
uint64_t nextHighestPowerOf2(uint64_t n);

#endif
