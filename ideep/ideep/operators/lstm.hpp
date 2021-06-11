#ifndef IDEEP_OPERATORS_LSTM_HPP
#define IDEEP_OPERATORS_LSTM_HPP

namespace ideep {

struct lstm_forward : public dnnl::lstm_forward {
   using super = dnnl::lstm_forward;

   static void compute(const tensor& src_layer,
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

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc, dst_iter_c_desc},
        aengine);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c}});
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
    auto src_iter_desc = src_iter.get_desc().to_type(src_layer.get_data_type());
    auto src_iter_c_desc = src_iter_c.get_desc();

    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    
    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, src_iter_desc, src_iter_c_desc},
         aengine);

    auto expected_weights_layer = pd.weights_layer_desc();
    auto expected_weights_iter = pd.weights_iter_desc();

    return std::make_tuple(expected_weights_layer, expected_weights_iter);
  } 
};

struct lstm_backward : public dnnl::lstm_backward {
  static void compute() {
  }
};

}  // namespace ideep

#endif