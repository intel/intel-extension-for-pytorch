#ifndef THSYCL_TENSORMATH_COMPARET_H
#define THSYCL_TENSORMATH_COMPARET_H
#include <THDP/THSYCLTensorMath.h>
#include <THDP/THSYCLTensor.hpp>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLNumerics.h>
#include <ATen/dpcpp/SYCLApplyUtils.h>

template <typename T, typename TOut>
struct TensorLTOp {
  inline void operator()(TOut &out, T &a, T &b) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::lt(a, b));
  }
};

template <typename T, typename TOut>
struct TensorGTOp {
  inline void operator()(TOut &out, T &a, T &b) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::gt(a, b));
  }
};

template <typename T, typename TOut>
struct TensorLEOp {
  inline void operator()(TOut &out, T &a, T &b) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::le(a, b));
  }
};

template <typename T, typename TOut>
struct TensorGEOp {
  inline void operator()(TOut &out, T &a, T &b) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::ge(a, b));
  }
};

template <typename T, typename TOut>
struct TensorEQOp {
  void operator()(TOut &out, T &a, T &b) const {
   out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::eq(a, b));
  }
};

template <typename T, typename TOut>
struct TensorNEOp {
  inline void operator()(TOut &out, T &a, T &b) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::ne(a, b));
  }
};



template<typename ScalarTypeOut, typename ScalarType, typename TensorTypeOut, typename TensorType, class Op> 
void THSYCL_logicalTensor(THSYCLState *state,
                       TensorTypeOut *self_,
                       TensorType *src1,
                       TensorType *src2,
                       Op op) {
  THSYCLTensor_resize(state, self_, src1->sizes(), {});

  THArgCheck(THSYCLTensor_nElement(state, src1) ==
             THSYCLTensor_nElement(state, src2), 3,
             "sizes do not match");
  at::sycl::SYCL_tensor_apply3<ScalarTypeOut, ScalarType, ScalarType>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), op);
 // if (!THC_pointwiseApply3<ScalarTypeOut, ScalarType, ScalarType>(state, self_, src1, src2, op)) {
   // THArgCheck(false, 2, SYCLTORCH_DIM_WARNING);
 // }

}


#endif
