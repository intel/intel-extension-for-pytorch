#include "AutoCast.hpp"
#include <torch/torch.h>

#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/function.h>

#include "AutoCast_utils.hpp"
#include "Common.hpp"
#include "Config.hpp"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/cpu/ExtendOPs.h"

namespace torch_ipex {
namespace autocast {
namespace int8 {

at::Tensor conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output =
        at::conv2d(input, weight, bias, stride, padding, dilation, groups);
    calibrate({input}, {weight}, {output}, "conv2d", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {weight},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::conv2d(
      r_inputs[0], r_weights[0], bias, stride, padding, dilation, groups);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor conv3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output =
        at::conv3d(input, weight, bias, stride, padding, dilation, groups);
    calibrate({input}, {weight}, {output}, "conv3d", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {weight},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::conv3d(
      r_inputs[0], r_weights[0], bias, stride, padding, dilation, groups);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = at::conv_transpose3d(
        input, weight, bias, stride, padding, output_padding, groups, dilation);
    calibrate(
        {input},
        {weight},
        {output},
        "conv_transpose3d",
        op_id,
        OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {weight},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::conv_transpose3d(
      r_inputs[0],
      r_weights[0],
      bias,
      stride,
      padding,
      output_padding,
      groups,
      dilation);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor _convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  // just need add id for int8 tracing path(conv2d -> conv2d -> _convolution).
  int64_t ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  return at::_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      benchmark,
      deterministic,
      cudnn_enabled,
      allow_tf32);
}

at::Tensor _convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled) {
  return at::_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      benchmark,
      deterministic,
      cudnn_enabled);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled);
    calibrate({input}, {}, {output}, "batch_norm", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::batch_norm(
      r_inputs[0],
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      cudnn_enabled);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::max_pool2d(
        input, kernel_size, stride, padding, dilation, ceil_mode);
    calibrate({input}, {}, {output}, "max_pool2d", op_id, OP_TYPE_POOLING);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::max_pool2d(
      r_inputs[0], kernel_size, stride, padding, dilation, ceil_mode);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor adaptive_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::adaptive_avg_pool2d(input, output_size);
    calibrate(
        {input}, {}, {output}, "adaptive_avg_pool2d", op_id, OP_TYPE_POOLING);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::adaptive_avg_pool2d(r_inputs[0], output_size);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor relu(const at::Tensor& input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::relu(input);
    calibrate({input}, {}, {output}, "relu", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::relu(r_inputs[0]);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor& relu_(at::Tensor& input) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto& output = at::relu_(input);
    calibrate({input}, {}, {output}, "relu_", op_id, OP_TYPE_INPLACE);
    return output;
  }
  return at::relu_(input);
}

at::Tensor sigmoid(const at::Tensor& input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::sigmoid(input);
    calibrate({input}, {}, {output}, "sigmoid", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::sigmoid(r_inputs[0]);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = at::linear(input, weight, bias);
    calibrate({input}, {weight}, {output}, "linear", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {weight},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::linear(r_inputs[0], r_weights[0], bias);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return output;
}

at::Tensor& add_tensor_(
    at::Tensor& input,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // if add has one scalar tensor or not float, bfloat16 dtype, we will not do
  // int8 path.
  if (input.ndimension() == 0 || other.ndimension() == 0 ||
      !(input.scalar_type() == at::kFloat ||
        input.scalar_type() == at::kBFloat16) ||
      !(other.scalar_type() == at::kFloat ||
        other.scalar_type() == at::kBFloat16)) {
    input.add_(other, alpha);
    return input;
  }
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    input.add_(other, alpha);
    calibrate({input, other}, {}, {input}, "add_", op_id, OP_TYPE_INPLACE);
    return input;
  }
  input.add_(other, alpha);
  return input;
}

at::Tensor add_tensor(
    const at::Tensor& input,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // if add has one scalar tensor or not float, bfloat16 dtype, we will not do
  // int8 path.
  if (input.ndimension() == 0 || other.ndimension() == 0 ||
      !(input.scalar_type() == at::kFloat ||
        input.scalar_type() == at::kBFloat16) ||
      !(other.scalar_type() == at::kFloat ||
        other.scalar_type() == at::kBFloat16)) {
    return at::add(input, other, alpha);
  }
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = at::add(input, other, alpha);
    calibrate({input, other}, {}, {output}, "add", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input, other},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::add(r_inputs[0], r_inputs[1], alpha);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = at::dropout(input, p, train);
    calibrate({input}, {}, {output}, "dropout", op_id, OP_TYPE_INPLACE);
    return output;
  }
  return at::dropout(input, p, train);
}

at::Tensor gelu(const at::Tensor& input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::gelu(input);
    calibrate({input}, {}, {output}, "gelu", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  return at::gelu(input);
}

std::tuple<Tensor, Tensor, Tensor> lstm(
    const Tensor& input,
    TensorList hx,
    TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  at::Tensor output, hy, cy;
  int step = has_biases ? 4 : 2;
  if (torch_ipex::check_int8_calibration()) {
    std::tie(output, hy, cy) = at::lstm(
        input,
        hx,
        _params,
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional,
        batch_first);
    std::vector<at::Tensor> weights;
    for (int i = 0; i < _params.size(); i += step) {
      auto w = at::cat({_params[i], _params[i + 1]}, 1);
      weights.push_back(w);
    }
    calibrate({input}, weights, {output}, "lstm", op_id, OP_TYPE_DEFAULT);
    return std::make_tuple(output, hy, cy);
  }
  params p = get_params(op_id);
  auto lstm_x = input;
  if (p.inputs_quantized[0]) {
    // add quantize and dequantize for input and weight.
    auto input_q = at::quantize_per_tensor(
        input,
        p.qparams[0][0].scale,
        p.qparams[0][0].zero_point,
        p.input_quantized_dtypes[0]);
    lstm_x = input_q.dequantize();
    std::vector<at::Tensor> lstm_params;
    if (torch_ipex::get_int8_weight_granularity(op_id) == "per_channel") {
      auto& weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
      for (int i = 0, idx = 0; i < _params.size(); i += step, idx += 1) {
        auto zero_points =
            at::zeros(weight_scale[idx].numel(), at::dtype(at::kLong));
        auto weight_q = at::quantize_per_channel(
            _params[i], weight_scale[idx], zero_points, 0, at::kQInt8);
        lstm_params.emplace_back(weight_q.dequantize());
        auto zero_points_2 =
            at::zeros(weight_scale[idx].numel(), at::dtype(at::kLong));
        auto weight_q_2 = at::quantize_per_channel(
            _params[i + 1], weight_scale[idx], zero_points_2, 0, at::kQInt8);
        lstm_params.emplace_back(weight_q_2.dequantize());
        if (has_biases) {
          lstm_params.emplace_back(_params[i + 2]);
          lstm_params.emplace_back(_params[i + 3]);
        }
      }
    } else {
      std::vector<float> w_scale = torch_ipex::get_int8_weight_scale(op_id);
      for (int i = 0, idx = 0; i < _params.size(); i += step, idx += 1) {
        auto weight_q =
            at::quantize_per_tensor(_params[i], w_scale[idx], 0, at::kQInt8);
        lstm_params.emplace_back(weight_q.dequantize());
        auto weight_q_2 = at::quantize_per_tensor(
            _params[i + 1], w_scale[idx], 0, at::kQInt8);
        lstm_params.emplace_back(weight_q_2.dequantize());
        if (has_biases) {
          lstm_params.emplace_back(_params[i + 2]);
          lstm_params.emplace_back(_params[i + 3]);
        }
      }
    }
    std::tie(output, hy, cy) = at::lstm(
        lstm_x,
        hx,
        lstm_params,
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional,
        batch_first);
  } else {
    std::tie(output, hy, cy) = at::lstm(
        lstm_x,
        hx,
        _params,
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional,
        batch_first);
  }
  // add quantize and dequantize output.
  if (p.outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(
        output,
        p.qparams[1][0].scale,
        p.qparams[1][0].zero_point,
        p.output_quantized_dtypes[0]);
    return std::make_tuple(output_q.dequantize(), hy, cy);
  }
  return std::make_tuple(output, hy, cy);
}

at::Tensor flatten(
    const at::Tensor& input,
    int64_t start_dim,
    int64_t end_dim) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto output = at::flatten(input, start_dim, end_dim);
    calibrate({input}, {}, {output}, "flatten", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {input},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::flatten(r_inputs[0], start_dim, end_dim);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::embedding_bag", "")
                       .typed<decltype(embedding_bag)>();
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output =
        op.call(weight, indices, offsets, sparse, include_last_offset);
    calibrate({}, {weight}, {output}, "embedding_bag", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  auto emb_w = weight;
  std::vector<float> w_scale;

  // For now embeddingbag will always quantized.
  // Input is "Long" type will not be quantzied, if next op support int8. Output
  // will not be quantized by default recipe.
  // TODO: How to indicate whether quantize embeddingbag ?
  TORCH_CHECK(torch_ipex::get_int8_weight_granularity(op_id) == "per_tensor");
  w_scale = torch_ipex::get_int8_weight_scale(op_id);
  auto weight_q = at::quantize_per_tensor(weight, w_scale[0], 0, at::kQInt8);
  emb_w = weight_q.dequantize();
  auto output = op.call(emb_w, indices, offsets, sparse, include_last_offset);
  // add quantize and dequantize output.
  if (p.outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, w_scale[0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor interaction_forward(const std::vector<at::Tensor>& input) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::interaction_forward", "")
                       .typed<decltype(interaction_forward)>();
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = op.call(input);
    calibrate(input, {}, {output}, "interaction", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);
  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      input,
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = op.call(r_inputs);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return outputs[0];
}

at::Tensor matmul(const at::Tensor& mat1, const at::Tensor& mat2) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto output = at::matmul(mat1, mat2);
    calibrate({mat1, mat2}, {}, {output}, "matmul", op_id, OP_TYPE_DEFAULT);
    return output;
  }
  params p = get_params(op_id);

  std::vector<at::Tensor> r_inputs, r_weights;
  std::tie(r_inputs, r_weights) = insert_q_dq_inputs(
      {mat1, mat2},
      {},
      p.qparams[0],
      p.input_quantized_dtypes,
      p.inputs_quantized,
      op_id);
  auto output = at::matmul(r_inputs[0], r_inputs[1]);
  auto outputs = insert_q_dq_outputs(
      {output}, p.qparams[1], p.output_quantized_dtypes, p.outputs_quantized);
  return output;
}

} // namespace int8
} // namespace autocast
} // namespace torch_ipex
