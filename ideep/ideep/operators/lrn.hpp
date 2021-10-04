#ifndef IDEEP_OPERATORS_LRN_HPP
#define IDEEP_OPERATORS_LRN_HPP

namespace ideep {

struct lrn_forward : public dnnl::lrn_forward {

  using super = dnnl::lrn_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels,
                      prop_kind aprop_kind = prop_kind::forward_training,
                      const engine& aengine = engine::cpu_engine()) {

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, local_size, alpha, beta, k},
        aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    exec_args args {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, dst}};

    bool with_workspace = aprop_kind == prop_kind::forward_training;
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct lrn_backward : public dnnl::lrn_backward {

  using super = dnnl::lrn_backward;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const tensor& dst,
                      tensor& diff_src,
                      dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels,
                      const engine& aengine = engine::cpu_engine()) {

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto forward_hints =
        lrn_forward::primitive_desc({prop_kind::forward_training, aalgorithm,
                                     src_desc, local_size, alpha, beta, k},
                                    aengine);

    auto pd = primitive_desc(
        {aalgorithm, src_desc, diff_dst.get_desc(), local_size, alpha, beta, k},
        aengine, forward_hints);
    
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    exec_args args {{DNNL_ARG_SRC, src},
                    {DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_DIFF_SRC, diff_src}};

    if (dst.has_workspace()) {
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif