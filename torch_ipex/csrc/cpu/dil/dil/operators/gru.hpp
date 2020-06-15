#ifndef DIL_OPERATORS_GRU_HPP
#define DIL_OPERATORS_GRU_HPP

namespace dil {

struct gru_forward : public dnnl::gru_forward {
  static void compute() {
  }
};

struct gru_backward : public dnnl::gru_backward {
  static void compute() {
  }
};

}  // namespace dil

#endif