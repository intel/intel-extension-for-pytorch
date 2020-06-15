#ifndef DIL_OPERATORS_VANILLA_RNN_HPP
#define DIL_OPERATORS_VANILLA_RNN_HPP

namespace dil {

/* 
 *
 * IMPORTANT NOTES(pinzhen): This is NOT A WORKING IMPLEMENTATION! It's just a 
 * sketch showing how to integrate dnnl to dil. Feel free to make any changes
 * to the interface design or class inheritance as you see fit.
 * 
 **/

struct rnn_forward : public dnnl::vanilla_rnn_forward {

  using super = dnnl::vanilla_rnn_forward;

  static void compute(
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const dims& dst_layer_dims,
      tensor& dst_layer,
      const dims& dst_iter_dims,
      tensor& dst_iter,
      tensor& workspace,
      rnn_kind akind,
      rnn_direction direction,
      prop_kind aprop_kind = prop_kind::forward_training,
      const engine& aengine = engine::cpu_engine()) {

    auto algo = utils::rnn_kind_to_algorithm(akind);
    auto activation = utils::rnn_kind_to_activation(akind);

    auto dtype = src_layer.get_data_type();
    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc().to_format_any();
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc().to_format_any();
    auto dst_layer_desc = tensor::desc(dst_layer_dims, dtype);
    auto dst_iter_desc = tensor::desc(dst_iter_dims, dtype, tag::any);

    auto pd = primitive_desc({
      aprop_kind,
      activation,
      direction,
      src_layer_desc,
      src_iter_desc,
      weights_layer_desc,
      weights_iter_desc,
      bias_desc,
      dst_layer_desc,
      dst_iter_desc,
    }, aengine);

    auto expected_src_layer = src_layer.reorder_if_differ_in(pd.src_layer_desc());
    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_layer_desc());
    auto expected_weight_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());

    // TOOD(xpz): query the dst desc from pd?
    dst_layer.reinit_if_possible(dst_layer_desc);
    dst_iter.reinit_if_possible(dst_iter_desc);

    exec_args args {{DNNL_ARG_SRC_LAYER, expected_src_layer},
                    {DNNL_ARG_SRC_ITER, expected_src_iter},
                    {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                    {DNNL_ARG_WEIGHTS_ITER, expected_weight_iter},
                    {DNNL_ARG_BIAS, expected_bias},
                    {DNNL_ARG_DST_LAYER, dst_layer},
                    {DNNL_ARG_DST_ITER, dst_iter}};

    if (aprop_kind == prop_kind::forward_training) {
      workspace.reinit_if_possible(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, workspace});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct rnn_backward : public dnnl::vanilla_rnn_backward {
  template <class alloc = utils::allocator>
  static void compute(
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const tensor& dst_layer,
      const tensor& dst_iter,
      const tensor& diff_dst_layer,
      const tensor& diff_dst_iter,
      const tensor& workspace,
      const bool with_bias,
      tensor& diff_src_layer,
      tensor& diff_src_iter,
      tensor& diff_weights_layer,
      tensor& diff_weights_iter,
      tensor& diff_bias,
      rnn_kind akind,
      rnn_direction direction,
      prop_kind aprop_kind = prop_kind::backward,
      const engine& aengine = engine::cpu_engine()) {
  }
};

}  // namespace dil

#endif
