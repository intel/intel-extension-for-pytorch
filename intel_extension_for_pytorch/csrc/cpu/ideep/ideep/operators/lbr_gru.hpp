#ifndef IDEEP_OPERATORS_LBR_GRU_HPP
#define IDEEP_OPERATORS_LBR_GRU_HPP

namespace ideep {

struct lbr_gru_forward : public dnnl::lbr_gru_forward {
  static void compute() {}
};

struct lbr_gru_backward : public dnnl::lbr_gru_backward {
  static void compute() {}
};

} // namespace ideep

#endif