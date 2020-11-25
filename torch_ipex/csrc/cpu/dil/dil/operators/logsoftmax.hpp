#ifndef DIL_OPERATORS_LOGSOFTMAX_HPP
#define DIL_OPERATORS_LOGSOFTMAX_HPP

namespace dil {

struct logsoftmax_forward : public dnnl::logsoftmax_forward {

  using super = dnnl::logsoftmax_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      int logsoftmax_axis,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_possible(src_desc);

    auto pd = primitive_desc(
        {aprop_kind, src_desc, logsoftmax_axis}, aengine);

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  }
};

struct logsoftmax_backward : public dnnl::logsoftmax_backward {

  using super = dnnl::logsoftmax_backward;

  static void compute(const tensor& dst,
                      const tensor& diff_dst,
                      tensor& diff_src,
                      int logsoftmax_axis,
                      const engine& aengine = engine::cpu_engine()) {

    auto dst_desc = dst.get_desc();
    // align data type of diff_dst with dst
    auto diff_dst_desc = diff_dst.get_desc().to_type(dst_desc.get_data_type());

    auto forward_hints = logsoftmax_forward::primitive_desc(
        {prop_kind::forward_inference, dst_desc, logsoftmax_axis}, aengine);

    auto pd =
        primitive_desc({diff_dst_desc, dst_desc, logsoftmax_axis},
                       aengine, forward_hints);
    auto expected_dst = dst.reorder_if_differ_in(pd.dst_desc());
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DST, expected_dst},
                       {DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
    
  }
};

}  // namespace dil

#endif