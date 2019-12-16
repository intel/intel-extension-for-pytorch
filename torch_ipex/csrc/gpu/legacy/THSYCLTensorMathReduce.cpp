#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensorCopy.h>
#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLStorage.hpp>
#include <legacy/THSYCLAlgorithm.h>
#include <legacy/THSYCLReduce.h>
#include <legacy/THSYCLReduceAll.h>
#include <legacy/THSYCLTensorMathReduce.h>

THSYCL_API int
THSyclByteTensor_logicalAndAll(THSYCLState *state, THSyclByteTensor *self) {
  THSYCLAssertSameGPU(THSyclByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THSYCL_reduceAll<uint8_t>(state, self,
                                 sycl_identity<unsigned char>(),
                                 LogicalAll(),
                                 (unsigned char) 1, &result, false)) {
    auto dims = THSyclByteTensor_nDimensionLegacyNoScalars(state, self);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 1, SYCLTORCH_DIM_WARNING);
  }

  return (int) result;
}

THSYCL_API int
THSyclByteTensor_logicalAnyAll(THSYCLState *state, THSyclByteTensor *self) {
  THSYCLAssertSameGPU(THSyclByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THSYCL_reduceAll<uint8_t>(state, self,
                                 sycl_identity<unsigned char>(),
                                 LogicalAny(),
                                 (unsigned char) 0, &result, false)) {
    auto dims = THSyclByteTensor_nDimensionLegacyNoScalars(state, self);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 1, SYCLTORCH_DIM_WARNING);
  }

  return (int) result;
}

THSYCL_API void
THSyclByteTensor_logicalAnd(THSYCLState* state, THSyclByteTensor *self, THSyclByteTensor *src, int dimension, int keepdim) {
  THSYCLAssertSameGPU(THSyclByteTensor_checkGPU(state, 2, self, src));
  if (!THSYCL_reduceDim<uint8_t>(state, self, src,
                                 sycl_identity<unsigned char>(),
                                 LogicalAll(),
                                 sycl_identity<unsigned char>(),
                                 (unsigned char) 1,
                                 dimension,
                                 keepdim)) {
    auto dims = THSyclByteTensor_nDimensionLegacyNoScalars(state, self);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 4, SYCLTORCH_DIM_WARNING);
  }

  c10::DeviceIndex currentDevice;
  THSYCLCheck(c10::sycl::syclGetDevice(&currentDevice));
}

THSYCL_API void
THSyclByteTensor_logicalAny(THSYCLState* state, THSyclByteTensor *self, THSyclByteTensor *src, int dimension, int keepdim) {
  THSYCLAssertSameGPU(THSyclByteTensor_checkGPU(state, 2, self, src));
  if (!THSYCL_reduceDim<uint8_t>(state, self, src,
                                 sycl_identity<unsigned char>(),
                                 LogicalAny(),
                                 sycl_identity<unsigned char>(),
                                 (unsigned char) 0,
                                 dimension,
                                 keepdim)) {
    auto dims = THSyclByteTensor_nDimensionLegacyNoScalars(state, self);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 4, SYCLTORCH_DIM_WARNING);
  }

  c10::DeviceIndex currentDevice;
  THSYCLCheck(c10::sycl::syclGetDevice(&currentDevice));
}

#include <legacy/generic/THSYCLTensorMathReduce.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathReduce.cpp>
#include <legacy/THSYCLGenerateBoolType.h>
