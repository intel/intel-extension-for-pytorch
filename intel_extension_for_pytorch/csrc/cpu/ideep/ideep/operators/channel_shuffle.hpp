#ifndef IDEEP_OPERATORS_CHANNEL_SHUFFLE_HPP
#define IDEEP_OPERATORS_CHANNEL_SHUFFLE_HPP

namespace ideep {

struct channel_shuffle_forward : public dnnl::shuffle_forward {
  using super = dnnl::shuffle_forward;

  static void compute(
      const tensor& src,
      tensor& dst,
      const int group,
      const int axis = 1,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(src.get_dim(axis) % group == 0, "Invalid channel and group");
    IDEEP_ENFORCE(src.get_data_type() == data_type::f32, "invalid data type");

    auto group_size = static_cast<int>(src.get_dim(axis) / group);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop_kind, src.get_desc(), axis, group_size}, aengine, op_attr);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_DST, dst},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

struct channel_shuffle_backward : public dnnl::shuffle_backward {
  using super = dnnl::shuffle_backward;

  static void compute(
      const tensor& diff_dst,
      tensor& diff_src,
      const int group,
      const int axis = 1,
      const engine& aengine = engine::cpu_engine()) {
    auto group_size = static_cast<int>(diff_dst.get_dim(axis) / group);
    auto data_desc = diff_dst.get_desc();

    auto forward_hints = dnnl::shuffle_forward::primitive_desc(
        {prop_kind::forward, data_desc, group_size, axis}, aengine);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {data_desc, axis, group_size}, aengine, forward_hints, op_attr);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
         {DNNL_ARG_DIFF_SRC, diff_src},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }
};

} // namespace ideep

#endif