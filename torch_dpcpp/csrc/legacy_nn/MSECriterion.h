#ifndef THDPNN_MSECRITERION_H
#define THDPNN_MSECRITERION_H

template <typename T>
struct TensorMSEOp {
  void operator()(T& out, T& in1, T& in2) const {
    out = (in1 - in2) * (in1 -in2);
  }
};

template <typename T>
struct TensorSubOp {
  void operator()(T& in1, T& in2) const {
    in1 = in1 -in2;
    in2 = in1;
    
  }
};

#endif