#ifndef IDEEP_OPERATORS_BINARY_HPP
#define IDEEP_OPERATORS_BINARY_HPP

namespace ideep {

struct binary : public dnnl::binary {

  using super = dnnl::binary;

  static void compute(const tensor& src0,
                      const tensor& src1,
                      tensor& dst,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src0_desc = src0.get_desc();
    auto src1_desc = src1.get_desc();
    auto dst_desc = src0_desc.to_format_any();

    auto pd = primitive_desc(
        {aalgorithm, src0_desc, src1_desc, dst_desc}, aengine);
    
    auto expected_src0 = src0.reorder_if_differ_in(pd.src0_desc());
    auto expected_src1 = src1.reorder_if_differ_in(pd.src1_desc());
    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_0, expected_src0},
                       {DNNL_ARG_SRC_1, expected_src1},
                       {DNNL_ARG_DST, dst}});
  }
};

}  // namespace ideep

#endif