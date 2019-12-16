#ifndef TH_SYCL_TENSOR_MATH_INC
#define TH_SYCL_TENSOR_MATH_INC

#include <legacy/THSYCLTensor.h>
#include <legacy/THSYCLGeneral.h>

#include <legacy/generic/THSYCLTensorMathBlas.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMath.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMath.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathScan.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMasked.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMasked.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathPairwise.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathPairwise.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathPointwise.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathPointwise.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathCompare.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathCompare.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathCompareT.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathCompareT.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMathReduce.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorMathReduce.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorSort.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorSort.h>
#include <legacy/THSYCLGenerateBoolType.h>

#include <legacy/generic/THSYCLTensorMode.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorTopK.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/THSYCLNumerics.h>

template<typename T>
class diag_to_sycl_ker {};

template<typename T>
class diag_from_sycl_ker {};

DP_DEF_K1(nonzero_kernel);

template<typename T>
struct NonZeroOp {
  NonZeroOp() {}
  bool operator()(T lhs) const {
    if(THSYCLNumerics<T>::ne(lhs, ScalarConvert<float, T>::to(0.0))) {
      return true;
    } else {
      return false;
    }
  }
};

THSYCL_API int THSyclByteTensor_logicalAndAll(THSYCLState *state, THSyclByteTensor *self);
THSYCL_API int THSyclByteTensor_logicalAnyAll(THSYCLState *state, THSyclByteTensor *self);

THSYCL_API void THSyclByteTensor_logicalAnd(THSYCLState* state, THSyclByteTensor *self, THSyclByteTensor *src, int dimension, int keepdim);
THSYCL_API void THSyclByteTensor_logicalAny(THSYCLState* state, THSyclByteTensor *self, THSyclByteTensor *src, int dimension, int keepdim);


#endif
