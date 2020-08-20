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
                      const scale_t& src_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy;
    compute_impl</*use_stats=*/false>(
        src, dummy, dummy, scale, shift, dst, epsilon, src_scales, dst_scales, flags, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*use_stats=*/true>(
        src, mean, variance, scale, shift, dst, epsilon, src_scales, dst_scales, flags, aengine);
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
                           const scale_t& src_scales,
                           const scale_t& dst_scales,
                           const batch_normalization_flag flags,
                           const engine& aengine) {
    auto pd_flags = batch_normalization_flag::use_scale_shift;
    if (use_stats)
      pd_flags |= batch_normalization_flag::use_global_stats;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    //auto src_desc = src._get_unblocked_desc_if_4c_blocked();
     auto src_desc = src.get_desc();

    bool fuse_norm_relu = (bool) (flags & batch_normalization_flag::fuse_norm_relu);
    attr_t attr = fuse_norm_relu ? attr_t::fuse_relu() : attr_t();
    auto pd = primitive_desc(
        {prop_kind::forward_inference, src_desc, epsilon, pd_flags}, attr, aengine);

  
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (!dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(dst_scales);
    }
 
    tensor scale_shift {pd.weights_desc()};
    if (src_scales.empty() && dst_scales.empty()) {
      auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
      std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
      std::memcpy(scale_shift_buf + scale.get_size(),
                  shift.get_data_handle(), shift.get_size());
    }

    if (use_stats) {
      tensor expected_mean, expected_var;
      // int8 path, need updata scale and shift
      if (!src_scales.empty() && (!dst_scales.empty())) {
        int channel_count = scale.get_nelems();
        float* scale_ptr = static_cast<float *>(scale.get_data_handle());
        float* shift_ptr = static_cast<float *>(shift.get_data_handle()); 
        float* scale_shift_buf = static_cast<float *>(scale_shift.get_data_handle());
 
        expected_mean = {pd.mean_desc()};
        expected_var = {pd.variance_desc()}; 
        float* mean_ptr = static_cast<float *>(mean.get_data_handle());
        float* var_ptr = static_cast<float *>(variance.get_data_handle());
        float* expected_mean_ptr = static_cast<float *>(expected_mean.get_data_handle());
        float* expected_var_ptr = static_cast<float *>(expected_var.get_data_handle());
    #ifdef _OPENMP
    #if (_OPENMP >= 201307)
    # pragma omp parallel for simd
    #else
    # pragma omp parallel for schedule(static)
    #endif
    #endif
        for (int c = 0; c < channel_count; c++) {
          float scale_temp = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon);
          scale_shift_buf[c] = scale_temp * dst_scales[0] / src_scales[0];
          scale_shift_buf[c + channel_count] = (shift_ptr[c] - mean_ptr[c] * scale_temp) * dst_scales[0];
          expected_mean_ptr[c] = 0.0f;
          expected_var_ptr[c] = 1.0f;
        }
      } else {
        expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
        expected_var = variance.reorder_if_differ_in(pd.variance_desc());
      }
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_SCALE_SHIFT, scale_shift},
                         {DNNL_ARG_VARIANCE, expected_var},
                         {DNNL_ARG_MEAN, expected_mean},
                         {DNNL_ARG_DST, dst}});
    } else {
      // int8 path, need updata scale and shift
      if (!src_scales.empty() && !dst_scales.empty()) {
        int channel_count = scale.get_nelems();
        float* scale_shift_buf = static_cast<float *>(scale_shift.get_data_handle());
        float* scale_ptr = static_cast<float *>(scale.get_data_handle());
        float* shift_ptr = static_cast<float *>(shift.get_data_handle());
        for (int c = 0; c < channel_count; c++) {
          scale_shift_buf[c] = scale_ptr[c] * dst_scales[0] / src_scales[0];
          scale_shift_buf[c + channel_count] = shift_ptr[c] * dst_scales[0];
        }
      }
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
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    auto pd_flags = flags | batch_normalization_flag::use_scale_shift;
    bool with_workspace = (bool) (flags & batch_normalization_flag::fuse_norm_relu);

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

    exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_SCALE_SHIFT, scale_shift},
                    {DNNL_ARG_MEAN, mean},
                    {DNNL_ARG_VARIANCE, variance},
                    {DNNL_ARG_DST, dst}};
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
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
                      float epsilon,
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift) {
   compute(src, scale, shift, dst, mean, variance, momentum, epsilon, flags);
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
                      const tensor& dst = tensor(),
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift,
                      const engine& aengine = engine::cpu_engine()) {
    // TODO: support no-affine model
    auto pd_flags = flags | batch_normalization_flag::use_scale_shift;
    bool with_workspace = (bool) (flags & batch_normalization_flag::fuse_norm_relu);
    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
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

    exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_SCALE_SHIFT, scale}, // only need scale
                    {DNNL_ARG_MEAN, expected_mean},
                    {DNNL_ARG_VARIANCE, expected_variance},
                    {DNNL_ARG_DIFF_SRC, diff_src},
                    {DNNL_ARG_DIFF_SCALE_SHIFT, diff_scale_shift}};
    if (with_workspace) {
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);   
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
                      const batch_normalization_flag flags = batch_normalization_flag::use_scale_shift,
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
