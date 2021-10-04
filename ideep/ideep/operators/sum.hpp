#ifndef IDEEP_OPERATORS_SUM_HPP
#define IDEEP_OPERATORS_SUM_HPP

namespace ideep {

struct sum : public dnnl::sum {

  using super = dnnl::sum;

  static void compute(const scale_t& scales,
                      const std::vector<tensor>& srcs,
                      tensor& dst,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_descs = utils::fmap(srcs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });
    auto pd = primitive_desc(scales, src_descs, aengine);

    dst.reinit_if_possible(pd.dst_desc());

    exec_args args {{DNNL_ARG_DST, dst}};
    for (int i = 0; i < srcs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif