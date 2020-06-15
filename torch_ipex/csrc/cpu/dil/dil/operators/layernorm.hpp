#ifndef DIL_OPERATORS_LAYERNORM_HPP
#define DIL_OPERATORS_LAYERNORM_HPP

namespace dil {

struct layer_normalization_forward : public dnnl::layer_normalization_forward {

  using super = dnnl::layer_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      float epsilon,
                      const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    auto pd_flags = flags | dnnl::normalization_flags::use_scale_shift;
    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto pd = primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, pd_flags}, aengine);

    tensor scale_shift {pd.weights_desc()};
    auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
    std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift_buf + scale.get_size(),
                shift.get_data_handle(), shift.get_size());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    mean.reinit_if_possible(pd.mean_desc());
    variance.reinit_if_possible(pd.variance_desc());
    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                       {DNNL_ARG_SCALE_SHIFT, scale_shift},
                       {DNNL_ARG_MEAN, mean},
                       {DNNL_ARG_VARIANCE, variance},
                       {DNNL_ARG_DST, dst}});
  }
};

struct layer_normalization_backward : public dnnl::layer_normalization_backward {

  using super = dnnl::layer_normalization_backward;

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& diff_dst,
                      const tensor& scale,
                      tensor& diff_src,
                      tensor& diff_scale_shift,
                      float epsilon,
                      const tensor& dst = tensor(),
                      const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    auto pd_flags = flags | dnnl::normalization_flags::use_scale_shift;
    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto forward_hints = dnnl::layer_normalization_forward::primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, pd_flags}, aengine);

    auto pd = primitive_desc(
        {prop_kind::backward, forward_hints.dst_desc(), src_desc, epsilon, pd_flags},
        aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
    auto expected_variance = variance.reorder_if_differ_in(pd.variance_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    diff_scale_shift.reinit_if_possible(pd.diff_weights_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                       {DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_SCALE_SHIFT, scale}, // only need scale
                       {DNNL_ARG_MEAN, expected_mean},
                       {DNNL_ARG_VARIANCE, expected_variance},
                       {DNNL_ARG_DIFF_SRC, diff_src},
                       {DNNL_ARG_DIFF_SCALE_SHIFT, diff_scale_shift}});   
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& diff_dst,
                      const tensor& scale,
                      tensor& diff_src,
                      tensor& diff_scale,
                      tensor& diff_shift,
                      float epsilon,
                      const tensor& dst = tensor(),
                      const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    tensor diff_scale_shift;
    compute(src, mean, variance, diff_dst, scale, diff_src, diff_scale_shift,
            epsilon, dst, flags, aengine);
    diff_scale.reinit_if_possible(scale.get_desc());
    diff_shift.reinit_if_possible(scale.get_desc());
    auto* diff_scale_shift_buf =
        static_cast<char*>(diff_scale_shift.get_data_handle());
    std::memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                diff_scale.get_size());
    std::memcpy(diff_shift.get_data_handle(),
                diff_scale_shift_buf + diff_scale.get_size(),
                diff_shift.get_size());
  }
};

}  // namespace dil

#endif
