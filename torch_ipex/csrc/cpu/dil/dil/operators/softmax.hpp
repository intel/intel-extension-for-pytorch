#ifndef DIL_OPERATORS_SOFTMAX_HPP
#define DIL_OPERATORS_SOFTMAX_HPP

namespace dil {

struct softmax_forward : public dnnl::softmax_forward {

  using super = dnnl::softmax_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      int softmax_axis,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = tensor::desc(src.get_dims(), src.get_data_type(), src.compute_strides(softmax_axis));
    dst.reinit_if_possible(src_desc);

    auto pd = primitive_desc(
        {aprop_kind, src_desc, softmax_axis}, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, dst}});
  }
};

struct softmax_backward : public dnnl::softmax_backward {

  using super = dnnl::softmax_backward;

  static void compute(const tensor& dst,
                      const tensor& diff_dst,
                      tensor& diff_src,
                      int softmax_axis,
                      const engine& aengine = engine::cpu_engine()) {

    auto dst_desc = tensor::desc(dst.get_dims(), dst.get_data_type(), dst.compute_strides(softmax_axis));

    auto forward_hints = softmax_forward::primitive_desc(
        {prop_kind::forward_inference, dst_desc, softmax_axis}, aengine);

    auto pd =
        primitive_desc({dst_desc, dst_desc, softmax_axis},
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