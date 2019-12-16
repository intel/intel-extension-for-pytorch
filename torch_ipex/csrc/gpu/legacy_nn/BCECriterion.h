#ifndef THDPNN_BCECRITERION_H
#define THDPNN_BCECRITERION_H

template <typename T>
inline T safe_log(T a) {
  if (a == 0.)
  {
    return THSYCLNumerics<T>::log((T)1e-12);
  }
  return THSYCLNumerics<T>::log(a);
}

template <typename T>
struct TensorBCEOp {
  void operator()(T& out, T& in, T& tar) const {
    out = -(safe_log(in) * tar + safe_log((T)1. - in) * ((T)1. - tar));
  }
};

template <typename T>
struct TensorLog1Op {
  void operator()(T& out, T& in) const {
    out = safe_log(in);
  }
};

template <typename T>
struct TensorLog2Op {
  void operator()(T& out, T& in) const {
    out = safe_log(1 - in);
  }
};

template <typename T>
struct TensorSub2Op {
  void operator()(T& out, T& in) const {
    out = 1 - in;
  }
};

#endif