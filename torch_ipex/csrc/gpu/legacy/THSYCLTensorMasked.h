#ifndef THSYCL_TENSOR_MASKED_H
#define THSYCL_TENSOR_MASKED_H
#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLNumerics.h>
#include <legacy/THSYCLAlgorithm.h>
#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLTensorMathReduce.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensorCopy.h>
#include <core/SYCLApplyUtils.h>
#include <legacy/generic/THSYCLTensorMasked.h>
#include <legacy/THSYCLGenerateAllTypes.h>

DP_DEF_K1(maskedSelect_scan_sycl_ker);
DP_DEF_K1(TensorMaskedSelectOp);
DP_DEF_K1(maskedCopy_scan_sycl_ker);
DP_DEF_K1(TensorMaskedCopyOp);

template <typename T, typename MaskT>
struct TensorMaskedFillOp {
  TensorMaskedFillOp(T v) : value(v) {}
  inline void operator()(T& t, MaskT& mask) const{
    if (mask) {
      t = value;
    }
  }

  T value;
};


#endif
