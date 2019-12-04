#ifndef TH_SYCL_TENSOR_MATH_INC
#define TH_SYCL_TENSOR_MATH_INC

#include <THDP/THSYCLTensor.h>
#include <THDP/THSYCLGeneral.h>

#include <THDP/generic/THSYCLTensorMathBlas.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMath.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMath.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathScan.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMasked.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMasked.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathPairwise.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathPairwise.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathPointwise.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathPointwise.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathCompare.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathCompare.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathCompareT.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathCompareT.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMathReduce.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorMathReduce.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorSort.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorSort.h>
#include <THDP/THSYCLGenerateBoolType.h>

#include <THDP/generic/THSYCLTensorMode.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorTopK.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/THSYCLNumerics.h>

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
