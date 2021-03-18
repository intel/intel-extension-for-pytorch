#ifndef DIL_OPERATORS_LSTM_HPP
#define DIL_OPERATORS_LSTM_HPP

namespace dil {

struct lstm_forward : public dnnl::lstm_forward {

  using super = dnnl::lstm_forward;

  static void compute(const dims& output_sizes,
                      const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      tensor& dst_layer,
                      tensor& dst_iter,
                      tensor& dst_iter_c,
                      const bool reverse = false,
                      prop_kind aprop = prop_kind::forward,
                      const std::vector<float>& data_scale = scale_t(),
                      const std::vector<int32_t>& data_shift = {},
                      const int weights_scale_mask = -1,
                      const std::vector<float>& weights_scales = scale_t(),
                      const engine& aengine = engine::cpu_engine()) {

    bool with_workspace = aprop == prop_kind::forward_training;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    auto src_layer_desc = src_layer.get_desc();

    auto src_iter_desc = src_iter.get_desc();
    // for fp32 and int8, src_iter_desc should be fp32
    // for bf16, src_iter_desc should be bf16
    if (src_layer.get_data_type() == data_type::bf16) {
      src_iter_desc = src_iter_desc.to_type(src_layer.get_data_type());
    }
    
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc();
    auto weights_iter_desc = weights_iter.get_desc();

    // If the weight is prepacked, the weight will be padded(fp32 & bf16) or blocked(int8), which is not dense
    // If not prepacked: use any format for weights
    // For accuracy consideration, weight remains fp32 when doing training,
    // so it is necessary to align weights data type with src in here.
    if (weights_layer_desc.is_dense()) {
      weights_layer_desc = weights_layer_desc.to_format_any().to_type(src_layer.get_data_type());
    }
    if (weights_iter_desc.is_dense()) {
      weights_iter_desc = weights_iter_desc.to_format_any().to_type(src_layer.get_data_type());
    }

    // When creating int8 LSTM pd, weight desc cannot be the prepacked s8 block format,
    // thus, we need to do to_format_any() here. 
    if (weights_layer_desc.get_data_type() == dil::data_type::s8) {
      weights_layer_desc = weights_layer_desc.to_format_any();
    }
    if (weights_iter_desc.get_data_type() == dil::data_type::s8) {
      weights_iter_desc = weights_iter_desc.to_format_any();
    }

    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);

    attr_t attr;
    DIL_ENFORCE((data_scale.size() == 1 && data_shift.size() == 1 && weights_scale_mask > -1 && !weights_scales.empty()) 
      || (data_scale.empty() && data_shift.empty() && weights_scale_mask == -1 && weights_scales.empty()), "Incorrect size for scale or zero point");
    
    if (!data_scale.empty()) {
      attr.set_rnn_data_qparams(data_scale[0], data_shift[0]);
      attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);
    }

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, src_iter_desc, src_iter_c_desc},
        attr, aengine);

    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_layer_desc(), attr);
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc(), attr);

    dst_layer.reinit_if_possible(pd.dst_layer_desc());
    dst_iter.reinit_if_possible(pd.dst_iter_desc());
    dst_iter_c.reinit_if_possible(pd.dst_iter_c_desc());

    exec_args args {{DNNL_ARG_SRC_LAYER, src_layer},
                    {DNNL_ARG_SRC_ITER, expected_src_iter},
                    {DNNL_ARG_SRC_ITER_C, src_iter_c},
                    {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                    {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                    {DNNL_ARG_BIAS, bias},
                    {DNNL_ARG_DST_LAYER, dst_layer},
                    {DNNL_ARG_DST_ITER, dst_iter},
                    {DNNL_ARG_DST_ITER_C, dst_iter_c}};

    if (with_workspace) {
      dst_layer.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst_layer.get_workspace()});
    }

    if (!data_scale.empty()) {
      dst_layer.set_scale(data_scale);
    }
    if (!data_shift.empty()) {
      dst_layer.set_zero_point(data_shift);
    }

    super(pd).execute(stream::default_stream(), args);
  }

  static std::tuple<tensor::desc, tensor::desc> expected_weights_desc(const dims& output_sizes,
                      const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const bool reverse = false,
                      prop_kind aprop = prop_kind::forward,
                      const std::vector<float>& data_scale = scale_t(),
                      const std::vector<int32_t>& data_shift = {},
                      const int weights_scale_mask = -1,
                      const std::vector<float>& weights_scales = scale_t(),
                      const engine& aengine = engine::cpu_engine()) {

    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    
    auto src_layer_desc = src_layer.get_desc();
    
    auto src_iter_desc = src_iter.get_desc();
    // for fp32 and int8, src_iter_desc should be fp32
    // for bf16, src_iter_desc should be bf16
    if (src_layer.get_data_type() == data_type::bf16) {
      src_iter_desc = src_iter_desc.to_type(src_layer.get_data_type());
    }
    
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    
    if (src_layer.get_data_type() == data_type::u8) {
      weights_layer_desc = weights_layer_desc.to_type(data_type::s8);
      weights_iter_desc = weights_iter_desc.to_type(data_type::s8);
    }

    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);

    attr_t attr;    
    DIL_ENFORCE((data_scale.size() == 1 && data_shift.size() == 1 && weights_scale_mask > -1 && !weights_scales.empty()) 
      || (data_scale.empty() && data_shift.empty() && weights_scale_mask == -1 && weights_scales.empty()), "Incorrect size for scale or zero point");
    
    if (!data_scale.empty()) {
      attr.set_rnn_data_qparams(data_scale[0], data_shift[0]);
      attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);
    }

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, src_iter_desc, src_iter_c_desc},
         attr, aengine);

    auto expected_weights_layer = pd.weights_layer_desc();
    auto expected_weights_iter = pd.weights_iter_desc();

    return std::make_tuple(expected_weights_layer, expected_weights_iter);
  }
};

struct lstm_backward : public dnnl::lstm_backward {

  using super = dnnl::lstm_backward;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& src_iter_c,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const tensor& dst_layer,
                      const tensor& dst_iter,
                      const tensor& dst_iter_c,
                      const tensor& diff_dst_layer,
                      const tensor& diff_dst_iter,
                      const tensor& diff_dst_iter_c,
                      tensor& diff_src_layer,
                      tensor& diff_src_iter,
                      tensor& diff_src_iter_c,
                      tensor& diff_weights_layer,
                      tensor& diff_weights_iter,
                      tensor& diff_bias,
                      const bool reverse = false,
                      const engine& aengine = engine::cpu_engine()) {
    auto aprop = prop_kind::backward;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc().to_type(src_layer.get_data_type());
    auto src_iter_c_desc = src_iter_c.get_desc();
    // use any format for weights
    // align weights data type with src
    auto weights_layer_desc = weights_layer.get_desc().to_format_any().to_type(src_layer.get_data_type());
    auto weights_iter_desc = weights_iter.get_desc().to_format_any().to_type(src_layer.get_data_type());
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto dst_iter_c_desc = dst_iter_c.get_desc();

    auto diff_src_layer_desc = src_layer_desc.to_type(data_type::f32);
    auto diff_src_iter_desc = src_iter_desc.to_type(data_type::f32);
    auto diff_src_iter_c_desc = src_iter_c_desc.to_type(data_type::f32);
    auto diff_weights_layer_desc = weights_layer_desc.to_type(data_type::f32);
    auto diff_weights_iter_desc = weights_iter_desc.to_type(data_type::f32);
    auto diff_bias_desc = bias_desc.to_type(data_type::f32);
    auto diff_dst_layer_desc = dst_layer_desc.to_type(data_type::f32);
    auto diff_dst_iter_desc = dst_iter_desc.to_type(data_type::f32);
    auto diff_dst_iter_c_desc = dst_iter_c_desc.to_type(data_type::f32);

    auto forward_hints =
        dnnl::lstm_forward::primitive_desc(
            {prop_kind::forward_training, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc, dst_iter_c_desc},
        aengine);

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc, dst_iter_c_desc,
         diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc,
         diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
         diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc},
        aengine, forward_hints);

    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    diff_src_layer.reinit_if_possible(pd.diff_src_layer_desc());
    diff_src_iter.reinit_if_possible(pd.diff_src_iter_desc());
    diff_src_iter_c.reinit_if_possible(pd.diff_src_iter_c_desc());
    //workaround: diff_weights_layer, diff_weights_iter and diff_bias need to clear before operation begin.
    diff_weights_layer.zero_init(pd.diff_weights_layer_desc());
    diff_weights_iter.zero_init(pd.diff_weights_iter_desc());
    diff_bias.zero_init(pd.diff_bias_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, expected_src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c},
                       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
                       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
                       {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c},
                       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer},
                       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter},
                       {DNNL_ARG_DIFF_BIAS, diff_bias},
                       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
                       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
                       {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c},
                       {DNNL_ARG_WORKSPACE, dst_layer.get_workspace()}});
  }
};

}  // namespace dil

#endif