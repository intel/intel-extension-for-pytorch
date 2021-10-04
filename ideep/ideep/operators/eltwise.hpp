#ifndef IDEEP_OPERATORS_ELTWISE_HPP
#define IDEEP_OPERATORS_ELTWISE_HPP

namespace ideep {

struct eltwise_forward : public dnnl::eltwise_forward {

  using super = dnnl::eltwise_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      algorithm aalgorithm = algorithm::eltwise_relu,
                      prop_kind aprop_kind = prop_kind::forward,
                      float alpha = 0.0,
                      float beta = 0.0,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_in = src;
    // we should leave dequantization to the framework
    if (aalgorithm != algorithm::eltwise_relu &&
        utils::one_of(src.get_data_type(), data_type::s8, data_type::u8)) {
      src_in = src_in.dequantize();
    }
    auto src_desc = src_in.get_desc();

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, alpha, beta}, aengine);

    dst.reinit_if_possible(pd.dst_desc());
    if (src_in.has_scale()) {
      dst.set_scale(src_in.get_scale());
    }

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src_in}, {DNNL_ARG_DST, dst}});

    // xpz: ???
    if (dst.has_scale() && aalgorithm == algorithm::eltwise_relu &&
        dst.get_data_type() == data_type::s8) {
      dst.to_type(data_type::u8);
    }
  }
};

struct eltwise_backward : public dnnl::eltwise_backward {
  using super = dnnl::eltwise_backward;
  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_src,
                      algorithm aalgorithm = algorithm::eltwise_relu,
                      float alpha = 0.0,
                      float beta = 0.0,
                      const engine& aengine = engine::cpu_engine()) {
  auto src_desc = src.get_desc();

  auto forward_hints = eltwise_forward::primitive_desc(
      {prop_kind::forward, aalgorithm, src_desc, alpha, beta}, aengine);
  auto pd =
      primitive_desc({aalgorithm, forward_hints.dst_desc(), src_desc, alpha, beta},
                     aengine, forward_hints);

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  diff_src.reinit_if_possible(pd.diff_src_desc());

  auto use_dst = utils::one_of(aalgorithm,
                               algorithm::eltwise_relu_use_dst_for_bwd,
                               algorithm::eltwise_tanh_use_dst_for_bwd,
                               algorithm::eltwise_elu_use_dst_for_bwd,
                               algorithm::eltwise_sqrt_use_dst_for_bwd,
                               algorithm::eltwise_logistic_use_dst_for_bwd,
                               algorithm::eltwise_exp_use_dst_for_bwd);
  auto src_dst_arg = use_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;

  super(pd).execute(stream::default_stream(),
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {src_dst_arg, expected_src},
                    {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};
}  // namespace ideep

#endif
