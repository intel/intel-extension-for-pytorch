#ifndef DIL_OPERATORS_BATCHNORM_HPP
#define DIL_OPERATORS_BATCHNORM_HPP
#include "sum.hpp"

namespace dil {

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward {

  using super = dnnl::batch_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy;
    compute_impl</*use_stats=*/false>(
        src, dummy, dummy, scale, shift, dst, epsilon, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*use_stats=*/true>(
        src, mean, variance, scale, shift, dst, epsilon, aengine);
  }
 private:
  template <bool use_stats>
  static void compute_impl(const tensor& src,
                           const tensor& mean,
                           const tensor& variance,
                           const tensor& scale,
                           const tensor& shift,
                           tensor& dst,
                           float epsilon,
                           const engine& aengine) {
    auto flags = batch_normalization_flag::use_scale_shift;
    if (use_stats)
      flags |= batch_normalization_flag::use_global_stats;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    auto pd = primitive_desc(
        {prop_kind::forward_inference, src_desc, epsilon, flags}, aengine);

    tensor scale_shift {pd.weights_desc()};
    auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
    std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift_buf + scale.get_size(),
                shift.get_data_handle(), shift.get_size());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (use_stats) {
      auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
      auto expected_var = variance.reorder_if_differ_in(pd.variance_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_SCALE_SHIFT, scale_shift},
                         {DNNL_ARG_VARIANCE, expected_var},
                         {DNNL_ARG_MEAN, expected_mean},
                         {DNNL_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_SCALE_SHIFT, scale_shift},
                         {DNNL_ARG_DST, dst}});
    }
  }
};

struct batch_normalization_forward_training
    : public dnnl::batch_normalization_forward {

  using super = dnnl::batch_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      float momentum,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    auto flags = batch_normalization_flag::use_scale_shift;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    auto pd = primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, flags}, aengine);

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

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      tensor& running_mean,
                      tensor& running_var,
                      float momentum,
                      float epsilon) {
   compute(src, scale, shift, dst, mean, variance, momentum, epsilon);
   dil::sum::compute({momentum, 1 - momentum}, {running_mean, mean},
                       running_mean);
   dil::sum::compute({momentum, 1 - momentum}, {running_var, variance},
                       running_var);
  }
};

struct batch_normalization_backward
    : public dnnl::batch_normalization_backward {

  using super = dnnl::batch_normalization_backward;

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& diff_dst,
                      const tensor& scale,
                      tensor& diff_src,
                      tensor& diff_scale_shift,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    // TODO: support no-affine model
    auto flags = batch_normalization_flag::use_scale_shift;
    auto src_desc = src.get_desc();
    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, flags}, aengine);
    auto diff_src_desc = diff_dst.get_desc();

    auto pd = primitive_desc(
        {prop_kind::backward, diff_src_desc, src_desc, epsilon, flags},
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
                      const engine& aengine = engine::cpu_engine()) {
  tensor diff_scale_shift;
  compute(src, mean, variance, diff_dst, scale, diff_src, diff_scale_shift,
          epsilon, aengine);
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
