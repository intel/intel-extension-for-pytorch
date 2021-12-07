#ifndef IDEEP_OPERATORS_GRU_HPP
#define IDEEP_OPERATORS_GRU_HPP

namespace ideep {

struct gru_forward : public dnnl::gru_forward {
  static void compute() {}
};

struct gru_backward : public dnnl::gru_backward {
  static void compute() {}
};

} // namespace ideep

#endif