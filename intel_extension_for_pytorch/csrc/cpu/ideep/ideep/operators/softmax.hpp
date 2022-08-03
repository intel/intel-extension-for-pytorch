#ifndef IDEEP_OPERATORS_SOFTMAX_HPP
#define IDEEP_OPERATORS_SOFTMAX_HPP

namespace ideep {

struct softmax_forward : public dnnl::softmax_forward {
  using super = dnnl::softmax_forward;

  static void compute(
      const tensor& src,
      tensor& dst,
      int softmax_axis,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_possible(src_desc);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd =
        primitive_desc({aprop_kind, src_desc, softmax_axis}, op_attr, aengine);
    tensor scratchpad(pd.scratchpad_desc());
    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC, src},
         {DNNL_ARG_DST, dst},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

} // namespace ideep

#endif