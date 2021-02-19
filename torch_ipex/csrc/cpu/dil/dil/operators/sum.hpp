#ifndef DIL_OPERATORS_SUM_HPP
#define DIL_OPERATORS_SUM_HPP

namespace dil {

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

    primitive_desc pd;
    if (srcs[0].shares_same_memory_with(dst)) {
      // in-place
      pd = primitive_desc(dst.get_desc(), scales, src_descs, aengine);
    } else {
      pd = primitive_desc(scales, src_descs, aengine);
      auto dst_desc = tensor::desc(pd.dst_desc(), srcs[0].get_groups());
      if (!dst_desc.is_dense()) {
        dst_desc = dst_desc.to_default_format();
        pd = primitive_desc(dst_desc, scales, src_descs, aengine);
      }
      // propagate src group info
      dst.reinit_if_possible(dst_desc);
    }

    exec_args args {{DNNL_ARG_DST, dst}};
    for (int i = 0; i < srcs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace dil

#endif