#ifndef DIL_OPERATORS_LSTM_HPP
#define DIL_OPERATORS_LSTM_HPP

namespace dil {

struct lstm_forward : public dnnl::lstm_forward {
  static void compute() {
  }
};

struct lstm_backward : public dnnl::lstm_backward {
  static void compute() {
  }
};

}  // namespace dil

#endif