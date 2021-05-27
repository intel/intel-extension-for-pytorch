#include "AutoCast.hpp"
#include <torch/torch.h>

#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/function.h>

#include "Common.hpp"
#include "Config.hpp"
#include "torch_ipex/csrc/autocast_mode.h"

namespace torch_ipex {
namespace autocast {
namespace int8 {

namespace {

using weakref_scales = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_scales = std::tuple<weakref_scales, at::Tensor>;
// TODO: zero_points cached
thread_local std::unordered_map<c10::TensorImpl *, val_scales> scales_casts;
// cach op tensors flow
// TODO: change the key work for slice or view.
using val_name = std::tuple<weakref_scales, std::string>;
thread_local std::unordered_map<c10::TensorImpl *, val_name> tensors_flow;

} // namespace

void clear_autocast_cache_int8() { scales_casts.clear(); }

at::Tensor conv2d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "conv2d." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::conv2d(input, weight, bias, stride, padding, dilation, groups);
    std::string op_output = "conv2d." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, weight,
                                          "conv2d", op_id, op_inputs, op_outputs);
    return output;
    }

    auto qparams = torch_ipex::get_int8_scales(op_id);
    std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
    std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
    std::vector<bool> inputs_quantized, outputs_quantized;
    std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
    auto conv_x = input;
    auto conv_w = weight;
    if (inputs_quantized[0]) {
      // add quantize and dequantize for input and weight.
      auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
          qparams[0][0].zero_point, input_quantized_dtypes[0]);
      conv_x = input_q.dequantize();
      if (torch_ipex::get_int8_weight_granularity(op_id) == "per_channel") {
        auto &weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
        auto zero_points = at::zeros(weight_scale.numel(), at::dtype(at::kLong));
        auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, at::kQInt8);
        conv_w = weight_q.dequantize();
      } else {
        float w_scale = torch_ipex::get_int8_weight_scale(op_id);
        auto weight_q = at::quantize_per_tensor(weight, w_scale, 0, at::kQInt8);
        conv_w = weight_q.dequantize();
      }
    }

    auto output = at::conv2d(conv_x, conv_w, bias, stride, padding, dilation, groups);
    // add quantize and dequantize output.
    if (outputs_quantized[0]) {
      auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
          qparams[1][0].zero_point, output_quantized_dtypes[0]);
      return output_q.dequantize();
    }
    return output;
}

at::Tensor conv3d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "conv3d." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::conv3d(input, weight, bias, stride, padding, dilation, groups);
    std::string op_output = "conv3d." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, weight,
                                          "conv3d", op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto conv_x = input;
  auto conv_w = weight;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input and weight.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    conv_x = input_q.dequantize();
    if (torch_ipex::get_int8_weight_granularity(op_id) == "per_channel") {
      auto &weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
      auto zero_points = at::zeros(weight_scale.numel(), at::dtype(at::kLong));
      auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, at::kQInt8);
      conv_w = weight_q.dequantize();
    } else {
      float w_scale = torch_ipex::get_int8_weight_scale(op_id);
      auto weight_q = at::quantize_per_tensor(weight, w_scale, 0, at::kQInt8);
      conv_w = weight_q.dequantize();
    }
  }

  auto output = at::conv3d(conv_x, conv_w, bias, stride, padding, dilation, groups);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor conv_transpose3d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "conv_transpose3d." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
    std::string op_output = "conv_transpose3d." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, weight,
                                          "conv_transpose3d", op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto conv_x = input;
  auto conv_w = weight;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input and weight.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    conv_x = input_q.dequantize();
    if (torch_ipex::get_int8_weight_granularity(op_id) == "per_channel") {
      auto &weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
      auto zero_points = at::zeros(weight_scale.numel(), at::dtype(at::kLong));
      auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, at::kQInt8);
      conv_w = weight_q.dequantize();
    } else {
      float w_scale = torch_ipex::get_int8_weight_scale(op_id);
      auto weight_q = at::quantize_per_tensor(weight, w_scale, 0, at::kQInt8);
      conv_w = weight_q.dequantize();
    }
  }

  auto output = at::conv_transpose3d(conv_x, conv_w, bias, stride, padding, output_padding, groups, dilation);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor _convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
    // just need add id for int8 tracing path(conv2d -> conv2d -> _convolution).
    int64_t ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
    return at::_convolution(input, weight, bias, stride, padding, dilation,
                            transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

at::Tensor _convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                        at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
                        bool transposed, at::IntArrayRef output_padding, int64_t groups,
                        bool benchmark, bool deterministic, bool cudnn_enabled) {
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

at::Tensor batch_norm(const at::Tensor &input, const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias,
     const c10::optional<at::Tensor> &running_mean, const c10::optional<at::Tensor> &running_var, bool training,
     double momentum, double eps, bool cudnn_enabled) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "batch_norm." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::batch_norm(input, weight, bias, running_mean, running_var,
                                 training, momentum, eps, cudnn_enabled);
    std::string op_output = "batch_norm." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, "batch_norm",
                                          op_id, op_inputs, op_outputs);
    return output;
  }
  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto bn_x = input;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    bn_x = input_q.dequantize();
  }
  auto output = at::batch_norm(bn_x, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor max_pool2d(const at::Tensor &input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "max_pool2d." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
    std::string op_output = "max_pool2d." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {input}, "max_pool2d",
                                          op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto pool_x = input;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    pool_x = input_q.dequantize();
  }
  auto output = at::max_pool2d(pool_x, kernel_size, stride, padding, dilation, ceil_mode);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor adaptive_avg_pool2d(const at::Tensor &input, at::IntArrayRef output_size) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "adaptive_avg_pool2d." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }

    auto output = at::adaptive_avg_pool2d(input, output_size);
    std::string op_output = "adaptive_avg_pool2d." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {input}, "adaptive_avg_pool2d",
                                          op_id, op_inputs, op_outputs);
    return output;
  }
  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto pool_x = input;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    pool_x = input_q.dequantize();
  }
  auto output = at::adaptive_avg_pool2d(pool_x, output_size);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor relu(const at::Tensor &input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "relu." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }

    auto output = at::relu(input);
    std::string op_output = "relu." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, "relu",
                                          op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto relu_x = input;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    relu_x = input_q.dequantize();
  }
  auto output = at::relu(relu_x);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor &relu_(at::Tensor &input) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "relu_." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto &output = at::relu_(input);
    std::string op_output = "relu_." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    if (it == tensors_flow.end()) {
      tensors_flow.emplace(output.unsafeGetTensorImpl(),
                           val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    } else {
      it->second = val_name{weakref_scales(output.getIntrusivePtr()), op_output};
    }
    torch_ipex::insert_or_updata_observer({}, {}, "relu_", op_id, op_inputs, op_outputs);
    return output;
  }

  return at::relu_(input);
}

at::Tensor sigmoid(const at::Tensor &input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "sigmoid." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }

    auto output = at::sigmoid(input);
    std::string op_output = "sigmoid." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, "sigmoid",
                                          op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto sigmoid_x = input;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    sigmoid_x = input_q.dequantize();
  }
  auto output = at::sigmoid(sigmoid_x);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor linear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "linear." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::linear(input, weight, bias);
    std::string op_output = "linear." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, weight,
                                          "linear", op_id, op_inputs, op_outputs);
    return output;
  }

  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto linear_x = input;
  auto linear_w = weight;
  if (inputs_quantized[0]) {
    // add quantize and dequantize for input and weight.
    auto input_q = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    linear_x = input_q.dequantize();
    if (torch_ipex::get_int8_weight_granularity(op_id) == "per_channel") {
      auto &weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
      auto zero_points = at::zeros(weight_scale.numel(), at::dtype(at::kLong));
      auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, at::kQInt8);
      linear_w = weight_q.dequantize();
    } else {
      float w_scale = torch_ipex::get_int8_weight_scale(op_id);
      auto weight_q = at::quantize_per_tensor(weight, w_scale, 0, at::kQInt8);
      linear_w = weight_q.dequantize();
    }
  }
  auto output = at::linear(linear_x, linear_w, bias);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor &add_tensor_(at::Tensor &input, const at::Tensor &other, const at::Scalar &alpha) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it1 = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it1 == tensors_flow.end()) {
      std::string op_input1 = "add_." + std::to_string(op_id) + ".input1";
      op_inputs.push_back(op_input1);
    } else {
      op_inputs.push_back(std::get<1>(it1->second));
    }
    auto it2 = tensors_flow.find(other.unsafeGetTensorImpl());
    if (it2 == tensors_flow.end()) {
      std::string op_input2 = "add_." + std::to_string(op_id) + ".input2";
      op_inputs.push_back(op_input2);
    } else {
      op_inputs.push_back(std::get<1>(it2->second));
    }

    input.add_(other, alpha);
    std::string op_output = "add_." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    if (it1 == tensors_flow.end()) {
      tensors_flow.emplace(input.unsafeGetTensorImpl(),
                           val_name{weakref_scales(input.getIntrusivePtr()), op_output});
    } else {
      it1->second = val_name{weakref_scales(input.getIntrusivePtr()), op_output};
    }
    torch_ipex::insert_or_updata_observer({}, {}, "add_", op_id, op_inputs, op_outputs);
    return input;
  }
  input.add_(other, alpha);
  return input;
}

at::Tensor add_tensor(const at::Tensor &input, const at::Tensor &other, const at::Scalar &alpha) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it1 = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it1 == tensors_flow.end()) {
      std::string op_input1 = "add." + std::to_string(op_id) + ".input1";
      op_inputs.push_back(op_input1);
    } else {
      op_inputs.push_back(std::get<1>(it1->second));
    }
    auto it2 = tensors_flow.find(other.unsafeGetTensorImpl());
    if (it2 == tensors_flow.end()) {
      std::string op_input2 = "add." + std::to_string(op_id) + ".input2";
      op_inputs.push_back(op_input2);
    } else {
      op_inputs.push_back(std::get<1>(it2->second));
    }

    auto output = at::add(input, other, alpha);
    std::string op_output = "add." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input, other}, {output}, "add", op_id, op_inputs, op_outputs);
    return output;
  }
  auto qparams = torch_ipex::get_int8_scales(op_id);
  std::vector<at::ScalarType> input_quantized_dtypes, output_quantized_dtypes;
  std::tie(input_quantized_dtypes, output_quantized_dtypes) = torch_ipex::get_int8_quantized_dtypes(op_id);
  std::vector<bool> inputs_quantized, outputs_quantized;
  std::tie(inputs_quantized, outputs_quantized) = torch_ipex::get_int8_insert_quantized_status(op_id);
  auto add_x1 = input;
  auto add_x2 = other;

  if (inputs_quantized[0]) {
    // add quantize and dequantize for input and weight.
    auto input_q1 = at::quantize_per_tensor(input, qparams[0][0].scale,
        qparams[0][0].zero_point, input_quantized_dtypes[0]);
    add_x1 = input_q1.dequantize();
  }

  if (inputs_quantized[1]) {
    // add quantize and dequantize for input and weight.
    auto input_q2 = at::quantize_per_tensor(other, qparams[0][1].scale,
        qparams[0][1].zero_point, input_quantized_dtypes[1]);
    add_x2 = input_q2.dequantize();
  }
  auto output = at::add(add_x1, add_x2, alpha);
  // add quantize and dequantize output.
  if (outputs_quantized[0]) {
    auto output_q = at::quantize_per_tensor(output, qparams[1][0].scale,
        qparams[1][0].zero_point, output_quantized_dtypes[0]);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor dropout(const at::Tensor &input, double p, bool train) {
  // always add id, but not compute scales or insert quant and dequant.
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (torch_ipex::check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "dropout." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto output = at::dropout(input, p, train);
    std::string op_output = "dropout." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    if (it == tensors_flow.end()) {
      tensors_flow.emplace(output.unsafeGetTensorImpl(),
                           val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    } else {
      it->second = val_name{weakref_scales(output.getIntrusivePtr()), op_output};
    }
    torch_ipex::insert_or_updata_observer({}, {}, "dropout", op_id, op_inputs, op_outputs);
    return output;
  }

  return at::dropout(input, p, train);
}

at::Tensor gelu(const at::Tensor &input) {
  auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  if (check_int8_calibration()) {
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "gelu." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }

    auto output = at::gelu(input);
    std::string op_output = "gelu." + std::to_string(op_id) + ".output";
    op_outputs.push_back(op_output);
    tensors_flow.emplace(output.unsafeGetTensorImpl(),
                         val_name{weakref_scales(output.getIntrusivePtr()), op_output});
    torch_ipex::insert_or_updata_observer({input}, {output}, "gelu",
                                          op_id, op_inputs, op_outputs);
    return output;
  }
}
  
} // namespace autocast
} // namespace cpu
} // namespace torch_ipex
