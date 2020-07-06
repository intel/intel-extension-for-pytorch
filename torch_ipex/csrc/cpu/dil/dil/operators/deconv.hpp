#ifndef DIL_OPERATORS_DECONV_HPP
#define DIL_OPERATORS_DECONV_HPP

namespace dil {

struct convolution_transpose_forward : public dnnl::deconvolution_forward {

  using super = dnnl::deconvolution_forward;

  static void compute(const tensor& src,
                      const tensor& weights, // dim: {i, o[, d], h, w}
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights, // dim: {i, o[, d], h, w}
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,   // [i, o, ...]
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const dims& src_dims = dims(),
      const dims& dst_dims = dims(),
      bool with_bias = true,
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    // (g)iohw -> (g)oihw
    std::swap(weights_dims_g[grouped + 0], weights_dims_g[grouped + 1]);
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto g = grouped ? dims_in[0] : 1;
    auto dilates_ = utils::get_compatible_dilates(dilates);

    dims x_dims, y_dims;
    if (src_dims.empty() || dst_dims.empty()) {
      auto ic = g * dims_in[1 + grouped];
      auto oc = g * dims_in[0 + grouped];
      auto kh = dims_in[ndims - 2];
      auto kw = dims_in[ndims - 1];
      auto h = 8 * kh;
      auto w = 8 * kw;
      auto oh = (h - 1) * strides[0] + (1 + (kh - 1) * (dilates[0])) - padding_l[0] - padding_r[0];
      auto ow = (w - 1) * strides[1] + (1 + (kw - 1) * (dilates[1])) - padding_l[1] - padding_r[1];

      x_dims = {1, ic, h, w};
      y_dims = {1, oc, oh, ow};
    } else {
      x_dims = src_dims;
      y_dims = dst_dims;
    }

    auto x_dtype = (dtype != data_type::s8) ? dtype : data_type::u8;
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    primitive_desc pd;
    // existence of bias may also affect the queried format (e.g. g8i32o64sp7k3)
    if (with_bias) {
      tensor::desc bias_desc({y_dims[1]}, y_dtype);
      pd = get_primitive_desc</*with_bias=*/true>(
          src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
          padding_l, padding_r, attr_t(), aalgorithm, aprop_kind);
    } else {
      pd = get_primitive_desc</*with_bias=*/false>(
          src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
          padding_l, padding_r, attr_t(), aalgorithm, aprop_kind);
    }

    // embed group info into weights_desc
    if (grouped) {
      // [g, o, i/g, ...] -> [g, i/g, o, ...]
      return tensor::desc(pd.weights_desc(), groups).transpose(1, 2);
    } else {
      // [o, i, ...] -> [i, o, ...]
      return tensor::desc(pd.weights_desc(), groups).transpose(0, 1);
    } 
  }

  template <bool with_bias>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::deconvolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_any = src_desc.to_format_any();
    auto weights_desc_any = weights_desc.to_format_any();
    auto bias_desc_any = with_bias ? bias_desc.to_format_any() : tensor::desc();
    auto dst_desc_any = dst_desc.to_format_any();

    if (with_bias) {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, bias_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    } else {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    }
  }

 private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           const dims& dst_dims,
                           tensor& dst,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           int groups,
                           const attr_t& attr,
                           algorithm aalgorithm,
                           prop_kind aprop_kind,
                           const engine& aengine) {

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups, true);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    tensor::desc dst_desc(dst_dims, src.get_data_type());

    auto pd = get_primitive_desc<with_bias>(
        src.get_desc(), weights_.get_desc(), bias.get_desc(), dst_desc,
        strides, dilates_, padding_l, padding_r, attr, aalgorithm,
        aprop_kind, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (with_bias) {
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst}});
    }
  }
};

struct convolution_transpose_backward_data
    : public dnnl::deconvolution_backward_data {

  using super = dnnl::deconvolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights, // dim: {i, o[, d], h, w}
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups, true);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();

    tensor::desc diff_src_desc(diff_src_dims, diff_dst_desc.get_data_type());

    auto forward_hints =
        convolution_transpose_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(), 
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};

struct convolution_transpose_backward_weights
    : public dnnl::deconvolution_backward_weights {

  using super = dnnl::deconvolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
   compute_impl</*with_diff_bias=*/true>(
       src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
       strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }
private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims, // [i, o, ...]
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);

    // dim: [i, o, ...]
    auto diff_weights_desc = 
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(), tag::any);

    if (groups > 1) {
      // dim: [g, o, i/g, ...]
      diff_weights_desc = diff_weights_desc.to_grouped(groups).transpose(1, 2);
    } else {
      // dim: [o, i, ...]
      diff_weights_desc = diff_weights_desc.transpose(0, 1);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto src_desc = src.get_desc().to_format_any();

    auto diff_bias_desc = with_diff_bias
        ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
              .to_format_any()
        : tensor::desc();

    auto forward_hints =
        convolution_transpose_forward::get_primitive_desc<with_diff_bias>(
            src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    // embed group info into diff_weights_desc
    auto expected_diff_weights_desc =
        tensor::desc(pd.diff_weights_desc(), groups);
    diff_weights.reinit_if_possible(expected_diff_weights_desc);

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                         {DNNL_ARG_DIFF_BIAS, diff_bias}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});
    }

    // recover output dims to align with pytorch
    if (groups > 1) {
      // [g, o, i/g, ...] -> [g, i/g, o, ...]
      diff_weights.transpose_(1, 2);
    } else {
      // [o, i, ...] -> [i, o, ...]
      diff_weights.transpose_(0, 1);
    }
  }
};
}  // namespace dil

#endif
