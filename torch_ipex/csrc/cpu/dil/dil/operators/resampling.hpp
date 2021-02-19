#ifndef DIL_OPERATORS_RESAMPLING_HPP
#define DIL_OPERATORS_RESAMPLING_HPP

namespace dil {

struct resampling_forward : public dnnl::resampling_forward {

  using super = dnnl::resampling_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      const dims& output_size,
                      const std::vector<float>& factors,
                      algorithm aalgorithm,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {

    auto src_desc = src.get_desc();
    auto dst_desc = src_desc.to_dims(output_size);

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, factors, src_desc, dst_desc}, aengine);

    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  }
};

struct resampling_backward : public dnnl::resampling_backward {

  using super = dnnl::resampling_backward;

  static void compute(const tensor& diff_dst,
                      tensor& diff_src,
                      const dims& input_size,
                      const std::vector<float>& factors,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {

    auto diff_dst_desc = diff_dst.get_desc();
    auto diff_src_desc = diff_dst_desc.to_dims(input_size);
    auto src_desc = diff_src_desc;
    auto dst_desc = diff_dst_desc;

    auto forward_hints = dnnl::resampling_forward::primitive_desc(
        {prop_kind::forward_training, aalgorithm, factors, src_desc, dst_desc}, aengine);

    auto pd =
        primitive_desc({aalgorithm, factors, diff_src_desc, diff_dst_desc},
                       aengine, forward_hints);

    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_SRC, diff_src},
                       {DNNL_ARG_DIFF_DST, diff_dst}});
    
  }
};

}  // namespace dil

#endif