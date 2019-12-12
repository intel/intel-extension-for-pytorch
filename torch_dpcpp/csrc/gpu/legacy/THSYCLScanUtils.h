#ifndef THSYCL_SCAN_UTILS_INC
#define THSYCL_SCAN_UTILS_INC
#include <legacy/THSYCLDeviceUtils.h>

// Collection of in-kernel scan / prefix sum utilities

// Inclusive Scan via an upsweep/downsweep mechanism. Assumes:
//
// 1. Power2ScanSize is a power of 2. This code still works for collections that
// do not exactly contain a power of 2 number of elements, simply round up to the
// nearest power of 2 and then call.
//
// 2. That there are two-elements per thread, i.e. the size of the smem storage
// is 2 * blockDim.x * sizeof(T).
//
// Consider a (+)-Scan on the following elements:
//
// Upsweep:
//
//    0  1  2  3  4  5  6  7
//       1     5     9    13
//             6          22
//                        28
//
// Downsweep:
//                  15
//         3     10    21
template <typename T>
using local_accessor_t = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

template <typename T, class BinaryOp, int Power2ScanSize>
void inclusivePrefixScan(T *smem, BinaryOp binop, const cl::sycl::nd_item<1> &item_id) {
  // Reduce step ("upsweep")
  int threadIdx = item_id.get_local_id(0);
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (threadIdx + 1) * stride * 2 - 1;
  if (index < Power2ScanSize) {
      smem[index] = binop(smem[index], smem[index - stride]);
    }
    item_id.barrier(cl::sycl::access::fence_space::local_space);
  }
// Post-reduce step ("downsweep")
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (threadIdx + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = binop(smem[index + stride], smem[index]);
    }
    item_id.barrier(cl::sycl::access::fence_space::local_space);
  }
}

// Inclusive prefix sum using shared Memory
template <typename T, class BinaryFunction>
void inclusivePrefixScan(const local_accessor_t<T>& smem, T in, T* out, 
    BinaryFunction binop, const cl::sycl::nd_item<1> &item_id) {
  // FIXME: this is slow
  int threadIdx = item_id.get_local_id(0);
  smem[threadIdx] = in;
  item_id.barrier(cl::sycl::access::fence_space::local_space);
  for (int64_t offset = 1; offset < static_cast<int64_t>(item_id.get_local_range(0)); offset *= 2) {
    T val = 0;

    if (threadIdx >= offset) {
      val = binop(smem[threadIdx - offset], smem[threadIdx]);
    }
    item_id.barrier(cl::sycl::access::fence_space::local_space);
      
    if (threadIdx >= offset) {
      smem[threadIdx] = val;
    }
    item_id.barrier(cl::sycl::access::fence_space::local_space);
  }
  *out = smem[threadIdx];
  item_id.barrier(cl::sycl::access::fence_space::local_space);
}

// Exclusive prefix sum using shared memory
template <typename T, class BinaryFunction>
void exclusivePrefixScan(const local_accessor_t<T>&smem, T in, T* out, T* carry, BinaryFunction binop, const cl::sycl::nd_item<1> &item_id) {
  // FIXME: crappy implementation
  // We kill write-after-read dependencies separately below, hence the `false`
  inclusivePrefixScan<T,  BinaryFunction>(smem, in, out, binop, item_id);

  *out -= in; 
  *carry = smem[item_id.get_local_range(0) - 1]; 

  // Prevent write-after-read dependencies on smem usage above if necessary
  item_id.barrier(cl::sycl::access::fence_space::local_space);
}



#endif
