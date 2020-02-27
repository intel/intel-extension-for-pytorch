#ifndef DIL_OPERATORS_DIRECT_COPY_HPP
#define DIL_OPERATORS_DIRECT_COPY_HPP

namespace dil {

struct direct_copy {
  static void compute(const tensor& src, tensor& dst) {
    dst.reinit_if_possible(src.get_desc());
    src.reorder_to(dst);
    if (src.has_scale()) {
      dst.set_scale(src.get_scale());
    }
  }
};

}  // namespace dil

#endif