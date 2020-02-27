#ifndef DIL_OPERATORS_CONV_HPP
#define DIL_OPERATORS_CONV_HPP

namespace dil {

struct convolution_forward_params {
  dnnl::convolution_forward::primitive_desc pd;
  // bias_attr contains requantization scales for bias
  attr_t bias_attr;
  scale_t dst_scales;
  int groups;
  tensor scratchpad;
};

struct convolution_forward : public dnnl::convolution_forward {

  using super = dnnl::convolution_forward;

  // prepare with bias
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    do_prepare</*with_bias=*/true>(
        param, src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // prepare without bias
  static void prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_prepare</*with_bias=*/false>(
        param, src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
  }

  // compute with bias
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst) {
    do_compute</*with_bias=*/true>(param, src, weights, bias, dst);
  }

  // compute without bias
  static void compute(const convolution_forward_params& param,
                      const tensor& src,
                      const tensor& weights,
                      tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false>(param, src, weights, dummy_bias, dst);
  }

  // 2-in-1 compute (prepare & compute) with bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    convolution_forward_params params;
    do_prepare</*with_bias=*/true>(
        params, src, weights, bias, dst_dims, dst, strides, dilates, 
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    do_compute</*with_bias=*/true>(params, src, weights, bias, dst);
  }

  // 2-in-1 compute (prepare & compute) without bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    convolution_forward_params params;
    do_prepare</*with_bias=*/false>(
        params, src, weights, dummy_bias, dst_dims, dst, strides, dilates, 
        padding_l, padding_r, groups, src_scales, weights_scales, dst_scales,
        attr, aalgorithm, aprop_kind, alowp_kind, aengine);
    do_compute</*with_bias=*/false>(params, src, weights, dummy_bias, dst);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {1, 1},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto grouped = groups > 1;
    auto weights_dims_g =
        grouped ? utils::group_dims(weights_dims, groups) : weights_dims;
    auto weights_desc = tensor::desc(weights_dims_g, dtype);

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto dilates_ = utils::get_compatible_dilates(dilates);

    DIL_ENFORCE(
        !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    auto ic = groups * dims_in[1 + grouped];
    auto oc = groups * dims_in[0 + grouped];
    auto kh = dims_in[ndims - 2];
    auto kw = dims_in[ndims - 1];
    int mb, h, w;
    if (src_dims.empty()) {
      // Construct a dummy case
      mb = 1;
      h = 2 * kh;
      w = 4 * kw;
    } else {
      // Use the real data
      mb = src_dims[0];
      h = src_dims[2];
      w = src_dims[3];
    }
    auto oh = (h - ((kh - 1) * (dilates_[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
    auto ow = (w - ((kw - 1) * (dilates_[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

    dims x_dims = { mb, ic, h, w };
    dims y_dims = { mb, oc, oh, ow };
    x_dtype = dtype == data_type::bf16 ? dtype : x_dtype;
    auto y_dtype = dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is
    // required by winograd primitive. Then, in the cases of variable input
    // shape, the detials of dnnl_wino_fmt will be changed. And, extra weihgts
    // reorder is inevitable each time, leading to bad performance. Here, we set
    // the prop_kind to forward, in order to reorder and cache weights as
    // blocked format, instead of dnnl_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd &&
        aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
        padding_l, padding_r, attr_t(), aalgorithm, apkind);

    // embed group info into weights_desc
    return tensor::desc(pd.weights_desc(), groups);
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
      algorithm aalgorithm = algorithm::convolution_direct,
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
  static void do_prepare(
      convolution_forward_params& param,
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales,
      const scale_t& weights_scales,
      const scale_t& dst_scales,
      const attr_t& attr,
      algorithm aalgorithm,
      prop_kind aprop_kind,
      const lowp_kind alowp_kind,
      const engine& aengine) {

    scale_t dst_scales_in;
    data_type dst_data_type;
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto& weights_scales_in =
        weights_.has_scale() ? weights_.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) {
      DIL_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : (src_scales.empty() ? DIL_DEF_SCALE : src_scales);

      // determine dst data type
      if (attr.has_op_kind(kind::sum)) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == DIL_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? DIL_DEF_SCALE
                          : dst_scales;

      scale_t bias_scales, op_scales;
      std::tie(bias_scales, op_scales) = utils::compute_scales(
          src_scales_in[0], dst_scales_in[0], weights_scales_in);

      if (attr.has_op_kind(kind::sum)) {
        float sum_scale =
            dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        if (attr.has_op_kind(kind::eltwise)) {
          op_attr = attr_t::residual(sum_scale);
        } else {
          op_attr = attr_t::fuse_sum(sum_scale);
        }
      } else if (attr.has_op_kind(kind::eltwise)) {
        op_attr = attr_t::fuse_relu();
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8, tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      weights_desc = weights_.get_desc().to_type(data_type::s8);
      if (weights_.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, groups > 1),
                        weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, tag::any};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                       bias_scales};
        }
      }
    } else {
      op_attr = attr;

      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      DIL_ENFORCE(utils::one_of(weights_.get_data_type(),
                                  data_type::f32, data_type::bf16),
                    "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = src.get_desc().to_format_any().to_type(dst_data_type);
      weights_desc = weights_.get_desc().to_format_any().to_type(dst_data_type);

      if (with_bias) {
        DIL_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc();
      }
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto dst_desc = attr.has_op_kind(kind::sum)
                        ? dst.get_desc()
                        : tensor::desc(dst_dims, dst_data_type);

    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc, strides, dilates_,
        padding_l, padding_r, op_attr, aalgorithm, aprop_kind, aengine);

    // allocate scratchpad
    tensor scratchpad(pd.scratchpad_desc());

    param = {pd, bias_attr, dst_scales, groups, scratchpad};
  }

  template <bool with_bias>
  static void do_compute(const convolution_forward_params& param,
                         const tensor& src, const tensor& weights,
                         const tensor& bias, tensor& dst) {
    auto& pd = param.pd;
    auto scratchpad = param.scratchpad;
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights.make_grouped_weights(param.groups)
                                .reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (!param.dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(param.dst_scales);
    }

    if (with_bias) {
      auto expected_bias =
          bias.reorder_if_differ_in(pd.bias_desc(), param.bias_attr);
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst},
                         {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }
  }
};


struct convolution_backward_data : public dnnl::convolution_backward_data {

  using super = dnnl::convolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    // align weight data type with diff_dst for bf16
    auto weights_desc =
        weights_.get_desc().to_format_any().to_type(diff_dst.get_data_type());

    tensor::desc diff_src_desc(diff_src_dims, diff_dst_desc.get_data_type());

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
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


struct convolution_backward_weights
    : public dnnl::convolution_backward_weights {

  using super = dnnl::convolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
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
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
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
                           const dims& diff_weights_dims,
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
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(), tag::any);
    if (groups > 1) {
        diff_weights_desc = diff_weights_desc.to_grouped(groups);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto src_desc = src.get_desc().to_format_any();

    // TODO: bf16 diff_bias
    auto diff_bias_desc =     
        tensor::desc({diff_dst.get_dim(1)}, data_type::f32, tag::any);

    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
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
  }
};
}  // namespace dil

#endif
