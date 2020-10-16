#ifndef DIL_OPERATORS_GRU_HPP
#define DIL_OPERATORS_GRU_HPP

namespace dil {

struct gru_forward : public dnnl::lbr_gru_forward {

  using super = dnnl::lbr_gru_forward;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      tensor& dst_layer,
                      tensor& dst_iter,
                      const bool reverse = false,
                      const engine& aengine = engine::cpu_engine()) {
    auto aprop = prop_kind::forward_inference;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc},
        aengine);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter}});
  }
};

struct gru_backward : public dnnl::lbr_gru_backward {

  using super = dnnl::lbr_gru_backward;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const tensor& dst_layer,
                      const tensor& dst_iter,
                      const tensor& diff_dst_layer,
                      const tensor& diff_dst_iter,
                      tensor& diff_src_layer,
                      tensor& diff_src_iter,
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
    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto diff_src_layer_desc = diff_src_layer.get_desc();
    auto diff_src_iter_desc = diff_src_iter.get_desc();
    auto diff_weights_layer_desc = diff_weights_layer.get_desc();
    auto diff_weights_iter_desc = diff_weights_iter.get_desc();
    auto diff_bias_desc = diff_bias.get_desc();
    auto diff_dst_layer_desc = diff_dst_layer.get_desc();
    auto diff_dst_iter_desc = diff_dst_iter.get_desc();

    auto forward_hints =
        dnnl::lbr_gru_forward::primitive_desc(
            {prop_kind::forward_inference, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc},
        aengine);

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc,
         diff_src_layer_desc, diff_src_iter_desc,
         diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
         diff_dst_layer_desc, diff_dst_iter_desc},
        aengine, forward_hints);
    
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
                       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
                       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer},
                       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter},
                       {DNNL_ARG_DIFF_BIAS, diff_bias},
                       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
                       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter}});
  }
};

}  // namespace dil

#endif