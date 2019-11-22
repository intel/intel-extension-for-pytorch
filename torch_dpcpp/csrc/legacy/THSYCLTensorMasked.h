#ifndef THSYCL_TENSOR_MASKED_H
#define THSYCL_TENSOR_MASKED_H
#include <THDP/THSYCLTensor.hpp>
#include <THDP/THSYCLNumerics.h>
#include <THDP/THSYCLAlgorithm.h>
#include <THDP/THSYCLTensorMath.h>
#include <THDP/THSYCLTensorMathReduce.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLTensorCopy.h>
#include <ATen/dpcpp/SYCLApplyUtils.h>
#include <THDP/generic/THSYCLTensorMasked.h>
#include <THDP/THSYCLGenerateAllTypes.h>

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
