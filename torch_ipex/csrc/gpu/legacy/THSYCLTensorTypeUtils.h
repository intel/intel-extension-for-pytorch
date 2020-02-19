#ifndef THSYCL_TENSOR_TYPE_UTILS_INC
#define THSYCL_TENSOR_TYPE_UTILS_INC

#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensor.hpp>

#include <core/detail/TensorInfo.h>

using namespace at::sycl::detail;

// Utility function for constructing TensorInfo structs. In this case, the
// two template parameters are:
//
// 1. The TensorType, e.g. THSYCLTensor in generic functions, or THSyclTensor,
// THSyclLongTensor etc.
//
// 2. The IndexType. This is always going to be an unsigned integral value,
// but depending on the size of the Tensor you may select uint16_t
// uint32_t, uint64_t etc.
//
// Internally we use the TensorUtils static functions to get the necessary
// dims, sizes, stride etc.
//
// For example, suppose we have a THSyclTensor t, with dim = 2, size = [3, 4],
// stride = [4, 1], offset = 8, and we set our index type to be unsigned int.
// Then we yield a TensorInfo struct templatized with float, unsigned int and
// the following fields:
//
// data is a float* to the underlying storage at position 8
// dims is 2
// sizes is a MAX_SYCLTORCH_DIMS element array with [3, 4] in its first two positions
// strides is a MAX_SYCLTORCH_DIMS element array with [4, 1] in its first two positions
//
// TensorInfos can then be passed to CUDA kernels, but we can use the static functions
// defined above to perform Tensor Operations that are appropriate for each
// TensorType.
template <typename ScalarType, typename TensorType, typename IndexType>
TensorInfo<ScalarType, IndexType>
getTensorInfo(THSYCLState* state, TensorType* t) {
  IndexType sz[MAX_TENSORINFO_DIMS];
  IndexType st[MAX_TENSORINFO_DIMS];

  int dims = THSYCLTensor_nDimensionLegacyNoScalars(state, t);
  for (int i = 0; i < dims; ++i) {
    sz[i] = THTensor_sizeLegacyNoScalars(t, i);
    st[i] = THTensor_strideLegacyNoScalars(t, i);
  }

  return TensorInfo<ScalarType, IndexType>(
    t->template data<ScalarType>(), dims, sz, st);
}

#endif
