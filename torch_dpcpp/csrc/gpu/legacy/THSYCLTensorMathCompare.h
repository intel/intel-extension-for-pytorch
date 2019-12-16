#ifndef THSYCL_TENSORMATH_COMPARE_H
#define THSYCL_TENSORMATH_COMPARE_H
#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLNumerics.h>
#include <core/SYCLApplyUtils.h>


template <typename T, typename TOut>
struct TensorLTValueOp {
  TensorLTValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::lt(in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
  TensorGTValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::gt(in, value));
  }

  const T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
  TensorLEValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::le(in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
  TensorGEValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::ge(in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
  TensorEQValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::eq(in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
  TensorNEValueOp(T v) : value(v) {}
  inline void operator()(TOut &out, T &in) const {
    out = ScalarConvert<bool, TOut>::to(THSYCLNumerics<T>::ne(in, value));
  }

  const T value;
};

template<typename ScalarTypeOut, typename ScalarType, typename TensorTypeOut, typename TensorType, class Op>
void THSYCL_logicalValue(THSYCLState *state,
                      TensorTypeOut *self_,
                      TensorType *src,
                      Op op) {
  THSYCLTensor_resize(state, self_, src->sizes(), {});
  at::sycl::SYCL_tensor_apply2<ScalarTypeOut, ScalarType>(THTensor_wrap(self_), THTensor_wrap(src), op);
}





#endif
