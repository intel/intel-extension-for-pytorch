#ifndef IDEEP_OPERATORS_LSTM_HPP
#define IDEEP_OPERATORS_LSTM_HPP

namespace ideep {

struct lstm_forward_inference : public dnnl::lstm_forward {
  using super = dnnl::lstm_forward;

  static void compute(
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
      const prop_kind aprop = prop_kind::forward_inference,
      const engine& aengine = engine::cpu_engine()) {
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    auto src_iter_c_desc = src_iter_c.get_desc();

    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();

    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto dst_iter_c_desc = dst_iter_c.get_desc();

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop,
         direction,
         src_layer_desc,
         src_iter_desc,
         src_iter_c_desc,
         weights_layer_desc,
         weights_iter_desc,
         bias_desc,
         dst_layer_desc,
         dst_iter_desc,
         dst_iter_c_desc},
        op_attr,
        aengine);

    auto expected_weights_layer =
        weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter =
        weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC_LAYER, src_layer},
         {DNNL_ARG_SRC_ITER, src_iter},
         {DNNL_ARG_SRC_ITER_C, src_iter_c},
         {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
         {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
         {DNNL_ARG_BIAS, bias},
         {DNNL_ARG_DST_LAYER, dst_layer},
         {DNNL_ARG_DST_ITER, dst_iter},
         {DNNL_ARG_DST_ITER_C, dst_iter_c},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }

  static std::tuple<tensor::desc, tensor::desc> expected_weights_desc(
      const dims& output_sizes,
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const bool reverse = false,
      prop_kind aprop = prop_kind::forward_inference,
      const engine& aengine = engine::cpu_engine()) {
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();

    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(
        output_sizes, src_layer.get_data_type(), tag::tnc);

    auto pd = primitive_desc(
        {aprop,
         direction,
         src_layer_desc,
         src_iter_desc,
         src_iter_c_desc,
         weights_layer_desc,
         weights_iter_desc,
         bias_desc,
         dst_layer_desc,
         src_iter_desc,
         src_iter_c_desc},
        aengine);

    auto expected_weights_layer = pd.weights_layer_desc();
    auto expected_weights_iter = pd.weights_iter_desc();

    return std::make_tuple(expected_weights_layer, expected_weights_iter);
  }
};

struct lstm_forward_training : public dnnl::lstm_forward {
  using super = dnnl::lstm_forward;

  static primitive_desc prepare(
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
      const engine& aengine = engine::cpu_engine()) {
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    auto src_iter_c_desc = src_iter_c.get_desc();
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto dst_iter_c_desc = dst_iter_c.get_desc();

    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {prop_kind::forward_training,
         direction,
         src_layer_desc,
         src_iter_desc,
         src_iter_c_desc,
         weights_layer_desc,
         weights_iter_desc,
         bias_desc,
         dst_layer_desc,
         dst_iter_desc,
         dst_iter_c_desc},
        op_attr,
        aengine);
    return pd;
  }

  static void compute(
      const primitive_desc& pd,
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const tensor& workspace,
      tensor& dst_layer,
      tensor& dst_iter,
      tensor& dst_iter_c,
      const bool reverse = false,
      const prop_kind aprop = prop_kind::forward_training,
      const engine& aengine = engine::cpu_engine()) {
    auto expected_weights_layer =
        weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter =
        weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    dst_layer.reinit_if_possible(pd.dst_layer_desc());
    dst_iter.reinit_if_possible(pd.dst_iter_desc());
    dst_iter_c.reinit_if_possible(pd.dst_iter_c_desc());
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC_LAYER, src_layer},
         {DNNL_ARG_SRC_ITER, src_iter},
         {DNNL_ARG_SRC_ITER_C, src_iter_c},
         {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
         {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
         {DNNL_ARG_BIAS, bias},
         {DNNL_ARG_DST_LAYER, dst_layer},
         {DNNL_ARG_DST_ITER, dst_iter},
         {DNNL_ARG_DST_ITER_C, dst_iter_c},
         {DNNL_ARG_WORKSPACE, workspace},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});
  }

  static std::tuple<tensor::desc, tensor::desc> expected_weights_desc(
      const dims& output_sizes,
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const bool reverse = false,
      prop_kind aprop = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();

    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(
        output_sizes, src_layer.get_data_type(), tag::tnc);

    auto pd = primitive_desc(
        {aprop,
         direction,
         src_layer_desc,
         src_iter_desc,
         src_iter_c_desc,
         weights_layer_desc,
         weights_iter_desc,
         bias_desc,
         dst_layer_desc,
         src_iter_desc,
         src_iter_c_desc},
        aengine);

    auto expected_weights_layer = pd.weights_layer_desc();
    auto expected_weights_iter = pd.weights_iter_desc();

    return std::make_tuple(expected_weights_layer, expected_weights_iter);
  }
};

struct lstm_backward : public dnnl::lstm_backward {
  using super = dnnl::lstm_backward;

  static void compute(
      const dnnl::lstm_forward::primitive_desc& forward_hints,
      const tensor& src_layer,
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
      const tensor& workspace,
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
    auto src_iter_desc = src_iter.get_desc();
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto dst_iter_c_desc = dst_iter_c.get_desc();

    // use any format for weights
    // align weights data type with src
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();

    auto diff_src_layer_desc = src_layer_desc.to_type(data_type::f32);
    auto diff_src_iter_desc = src_iter_desc.to_type(data_type::f32);
    auto diff_src_iter_c_desc = src_iter_c_desc.to_type(data_type::f32);
    auto diff_weights_layer_desc = weights_layer_desc.to_type(data_type::f32);
    auto diff_weights_iter_desc = weights_iter_desc.to_type(data_type::f32);
    auto diff_bias_desc = bias_desc.to_type(data_type::f32);
    auto diff_dst_layer_desc = dst_layer_desc.to_type(data_type::f32);
    auto diff_dst_iter_desc = dst_iter_desc.to_type(data_type::f32);
    auto diff_dst_iter_c_desc = dst_iter_c_desc.to_type(data_type::f32);

    // Use user mode scratchpad
    auto op_attr = dnnl::primitive_attr();
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = primitive_desc(
        {aprop,
         direction,
         src_layer_desc,
         src_iter_desc,
         src_iter_c_desc,
         weights_layer_desc,
         weights_iter_desc,
         bias_desc,
         dst_layer_desc,
         dst_iter_desc,
         dst_iter_c_desc,
         diff_src_layer_desc,
         diff_src_iter_desc,
         diff_src_iter_c_desc,
         diff_weights_layer_desc,
         diff_weights_iter_desc,
         diff_bias_desc,
         diff_dst_layer_desc,
         diff_dst_iter_desc,
         diff_dst_iter_c_desc},
        op_attr,
        aengine,
        forward_hints);

    auto expected_weights_layer =
        weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weights_iter =
        weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    diff_src_layer.reinit_if_possible(pd.diff_src_layer_desc());
    diff_src_iter.reinit_if_possible(pd.diff_src_iter_desc());
    diff_src_iter_c.reinit_if_possible(pd.diff_src_iter_c_desc());

    // workaround: diff_weights_layer, diff_weights_iter and diff_bias need to
    // clear before operation begin.
    tensor expected_diff_weights_layer;
    expected_diff_weights_layer.zero_init(pd.diff_weights_layer_desc());
    tensor expected_diff_weights_iter;
    expected_diff_weights_iter.zero_init(pd.diff_weights_iter_desc());
    tensor expected_diff_bias;
    expected_diff_bias.zero_init(pd.diff_bias_desc());
    tensor scratchpad(pd.scratchpad_desc());

    super(pd).execute(
        stream::default_stream(),
        {{DNNL_ARG_SRC_LAYER, src_layer},
         {DNNL_ARG_SRC_ITER, src_iter},
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
         {DNNL_ARG_DIFF_WEIGHTS_LAYER, expected_diff_weights_layer},
         {DNNL_ARG_DIFF_WEIGHTS_ITER, expected_diff_weights_iter},
         {DNNL_ARG_DIFF_BIAS, expected_diff_bias},
         {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
         {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
         {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c},
         {DNNL_ARG_WORKSPACE, workspace},
         {DNNL_ARG_SCRATCHPAD, scratchpad}});

    diff_weights_layer.feed_from(expected_diff_weights_layer);
    diff_weights_iter.feed_from(expected_diff_weights_iter);
    diff_bias.feed_from(expected_diff_bias);
  }
};

} // namespace ideep

#endif
