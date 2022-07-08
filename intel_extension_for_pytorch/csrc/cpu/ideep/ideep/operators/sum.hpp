#ifndef IDEEP_OPERATORS_SUM_HPP
#define IDEEP_OPERATORS_SUM_HPP

namespace ideep {

struct sum : public dnnl::sum {
  using super = dnnl::sum;

  static void compute(
      const scale_t& scales,
      const std::vector<tensor>& srcs,
      tensor& dst,
      const engine& aengine = engine::cpu_engine()) {
    auto src_descs = utils::fmap(srcs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(scales, src_descs, aengine, op_attr);

    dst.reinit_if_possible(pd.dst_desc());
    tensor scratchpad(pd.scratchpad_desc());
    exec_args args{{DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratchpad}};
    for (int i = 0; i < srcs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

} // namespace ideep

#endif
