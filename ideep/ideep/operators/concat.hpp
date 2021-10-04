#ifndef IDEEP_OPERATORS_CONCAT_HPP
#define IDEEP_OPERATORS_CONCAT_HPP

namespace ideep {

struct concat : public dnnl::concat {

  using super = dnnl::concat;

  static void compute(const std::vector<tensor>& inputs,
                      int axis,
                      tensor& output,
                      const engine& aengine = engine::cpu_engine()) {
    auto input_descs = utils::fmap(inputs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });

    // create a pd to query the optimimal format for src and dst
    auto pd = primitive_desc(axis, input_descs, aengine);
    auto expected_desc = tensor::desc(pd.dst_desc());

    output.reinit_if_possible(expected_desc);

    exec_args args {{DNNL_ARG_DST, output}};

    // DNNL currently supports two types of implementations in the concat:
    //   (Very fast) Works only when all memories are in the same format
    //   (Slower) Generic one, based on reorders: concat of n tensors is a set
    //            of n reorders from input to the proper part of the output
    // In case you have only two inputs there should not be performance 
    // difference between reordering one input to the format of the other one 
    // and emit the fast concat implementation versus using generic concat which
    // emits two reorders. So we align all tensors to the same optimial format
    // only when there are more than two inputs.
    auto opt_inputs = inputs;
    if (inputs.size() > 2) {
      opt_inputs = utils::fmap(inputs, [&](const tensor& t) {
        // construct a desc with dims of t and keep expected blocking format
        auto desc = expected_desc.to_dims(t.get_dims());
        // then reorder t to expected format if necessary
        return t.reorder_if_differ_in(desc);
      });
      input_descs = utils::fmap(opt_inputs, [](const tensor& t) {
        return static_cast<memory::desc>(t.get_desc());
      });
      // recreate the pd on new inputs with same formats
      pd = primitive_desc(axis, input_descs, aengine);
    }

    for (int i = 0; i < opt_inputs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, opt_inputs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }

  // for caffe2
  static std::vector<int32_t> compute(
      std::vector<tensor>& inputs,
      int axis,
      bool add_axis,
      tensor& dst,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(axis < (inputs[0].ndims() + add_axis),
                  "invalid axis in concat");
    for (int i = 0; i < inputs[0].ndims(); i++) {
      if (i == axis && !add_axis) continue;
      for (unsigned j = 1; j < inputs.size(); j++) {
        IDEEP_ENFORCE(inputs[j].get_dim(i) == inputs[0].get_dim(i),
                      "invalid input dims in concat");
      }
    }

    int32_t dst_channels = 0;
    std::vector<int32_t> axis_info(inputs.size(), 0);
    for (unsigned k = 0; k < inputs.size(); k++) {
      axis_info[k] = add_axis ? 1 : inputs[k].get_dim(axis);
      dst_channels += axis_info[k];
    }

    dims dst_dims(inputs[0].get_dims());
    if (add_axis)
      dst_dims.insert(dst_dims.begin() + axis, dst_channels);
    else
      dst_dims[axis] = dst_channels;

    auto dst_data_type = inputs[0].get_data_type();
    scale_t min_scale(IDEEP_DEF_SCALE);
    if (utils::one_of(dst_data_type, data_type::s8, data_type::u8)) {
      min_scale[0] = std::numeric_limits<float>::max();
      for (auto i : inputs) {
        if (i.get_data_type() != dst_data_type) {
          min_scale = IDEEP_DEF_SCALE;
          dst_data_type = data_type::f32;
          break;
        }
        if (i.has_scale() && (min_scale[0] > i.get_scale()[0])) {
          IDEEP_ENFORCE(i.get_scale().size() == 1, "incorrect scale size");
          min_scale[0] = i.get_scale()[0];
        }
      }
    }

    dims offset_dims(dst_dims.size(), 0);
    if (add_axis) {
      dst.reinit_if_possible({dst_dims, dst_data_type});
    } else {
      // construct dst tensor with dst_dims while keeping the same
      // blocking format as inputs[0]
      auto dst_desc = inputs[0].get_desc().to_dims(dst_dims);
      dst.reinit_if_possible(dst_desc);
    }
      
    if (utils::one_of(dst_data_type, data_type::s8, data_type::u8))
      dst.set_scale(min_scale);

    scale_t scales(1);
    // FIXME: To avoid view issue in dnnl
    // NOTE: In dnnl concat, dim 3 and 6+ are not supported.
    // Morewhile, the tensor shape must be blockable to create a view.
    if (!add_axis && dst_dims.size() != 3 && dst_dims.size() < 6) {
      for (unsigned k = 0; k < inputs.size(); k++) {
        if (!inputs[k].get_desc().is_limited_blockable()) {
          for (int i = 0; i < inputs.size(); ++i) {
            float input_scale =
                inputs[i].has_scale() ? inputs[i].get_scale()[0] : 1.0f;
            if (inputs[i].get_data_type() != dst_data_type ||
                input_scale - min_scale[0] != 0) {
              scales[0] = min_scale[0] / input_scale;
              tensor input_fp(inputs[i].get_desc().to_type(dst_data_type));
              inputs[i].reorder_to(input_fp, {0, scales});
              inputs[i] = input_fp;
            }
          }
          compute(inputs, axis, dst);
          return axis_info;
        }
      }
    }

    for (unsigned i = 0; i < inputs.size(); ++i) {
      auto input_i = inputs[i];
      auto in_dims = inputs[i].get_dims();
      auto in_scales = input_i.has_scale() ? input_i.get_scale()[0] : 1.0;
      scales[0] = min_scale[0] / in_scales;
      if (add_axis) {
        in_dims.insert(in_dims.begin() + axis, 1);
        input_i = input_i.reshape(in_dims);
      }
      dst.insert_submemory(input_i, in_dims, offset_dims, {0, scales});
      offset_dims[axis] += axis_info[i];
    }

    return axis_info;
  }
};

}  // namespace ideep

#endif