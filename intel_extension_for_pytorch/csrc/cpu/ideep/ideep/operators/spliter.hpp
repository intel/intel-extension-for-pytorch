#ifndef IDEEP_OPERATORS_SPLITER_HPP
#define IDEEP_OPERATORS_SPLITER_HPP

namespace ideep {

struct spliter {
  static std::vector<tensor> compute(
      const tensor& input,
      std::vector<int32_t>& axis_info,
      int axis,
      bool add_axis = false) {
    std::vector<tensor> outputs;
    tensor::dims output_dims(input.get_dims());
    tensor::dims offset_dims(output_dims.size(), 0);
    IDEEP_ENFORCE(axis < input.ndims(), "invalid axis in split");

    for (auto i = 0; i < axis_info.size(); ++i) {
      output_dims[axis] = axis_info[i];
      auto output = input.extract_submemory(output_dims, offset_dims);

      if (input.has_scale()) {
        output.set_scale(input.get_scale());
      }

      if (add_axis) {
        tensor::dims out_dims(output_dims);
        out_dims.erase(out_dims.begin() + axis);
        output.reshape(out_dims);
      }

      outputs.emplace_back(output);
      offset_dims[axis] += axis_info[i];
    }

    return outputs;
  }
};

} // namespace ideep

#endif