#ifndef IDEEP_OPERATORS_POOL_HPP
#define IDEEP_OPERATORS_POOL_HPP

namespace ideep {

struct pooling_forward : public dnnl::pooling_forward {
  using super = dnnl::pooling_forward;

  static void compute(
      const tensor& src,
      const dims& output_sizes,
      tensor& dst,
      const dims& strides,
      const dims& kernel,
      const dims& padding_l,
      const dims& padding_r,
      algorithm aalgorithm,
      prop_kind aprop_kind = prop_kind::forward_inference,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();

    tensor::desc dst_desc(output_sizes, src.get_data_type(), tag::any);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop_kind,
         aalgorithm,
         src_desc,
         dst_desc,
         strides,
         kernel,
         padding_l,
         padding_r},
        op_attr,
        aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args{
        {DNNL_ARG_SRC, expected_src},
        {DNNL_ARG_DST, dst},
        {DNNL_ARG_SCRATCHPAD, scratchpad}};
    super(pd).execute(stream::default_stream(), args);
  }
};

} // namespace ideep

#endif