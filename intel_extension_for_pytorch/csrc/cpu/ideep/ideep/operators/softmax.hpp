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

struct softmax_backward : public dnnl::softmax_backward {
  using super = dnnl::softmax_backward;

  static void compute(
      const tensor& dst,
      const tensor& diff_dst,
      tensor& diff_src,
      int softmax_axis,
      const engine& aengine = engine::cpu_engine()) {
    auto forward_hints = softmax_forward::primitive_desc(
        {prop_kind::forward_inference, dst.get_desc(), softmax_axis}, aengine);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {diff_dst.get_desc(), dst.get_desc(), softmax_axis},
        op_attr,
        aengine,
        forward_hints);
    auto expected_dst = dst.reorder_if_differ_in(pd.dst_desc());
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_DST, expected_dst},
         {DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_DIFF_SRC, diff_src},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

} // namespace ideep

#endif