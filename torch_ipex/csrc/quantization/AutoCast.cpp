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

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2; 
    }
};

using weakref_scales = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_scales = std::tuple<weakref_scales, at::Tensor>;
// TODO: zero_points cached
thread_local std::unordered_map<c10::TensorImpl *, val_scales> scales_casts;
// cach op tensors flow
// TODO: change the key work for slice or view.
using val_name = std::tuple<weakref_scales, std::string>;
thread_local std::unordered_map<c10::TensorImpl *, val_name> tensors_flow;

}  // namespace

void clear_autocast_cache_int8() { scales_casts.clear(); }

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  if (torch_ipex::check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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

  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales(
    {false}, {false}, num_ops_id);
  std::vector<float> w_scales = torch_ipex::get_int8_weight_scales(num_ops_id);

  if (!quantized) {
    return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto conv_x = input;
  auto conv_w = weight;
  if (pre_quantized) {
    // add quantize and dequantize for input and weight.
    const auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    conv_x = input_q.dequantize();
    if (w_scales.size() == 1) {
      auto weight_q = at::quantize_per_tensor(weight, w_scales[0], 0, at::kQInt8);
      conv_w = weight_q.dequantize();
    } else {
      // first cache the scales for imperative path.
      if (scales_casts.find(weight.unsafeGetTensorImpl()) == scales_casts.end()) {
        const auto casted_scale = at::tensor(w_scales, at::device(at::kCPU).dtype(at::kDouble));
        scales_casts.emplace(weight.unsafeGetTensorImpl(),
            val_scales{weakref_scales(weight.getIntrusivePtr()), casted_scale});
      }
      auto it = scales_casts.find(weight.unsafeGetTensorImpl());
      auto weight_scale = std::get<1>(it->second);
      auto zero_points = at::zeros(w_scales.size(), at::dtype(at::kLong));
      auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, c10::kQInt8);
      conv_w = weight_q.dequantize();
    }
  }

  auto output = at::conv2d(conv_x, conv_w, bias, stride, padding, dilation, groups);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  // just need add id for int8 tracing path(conv2d -> conv2d -> _convolution).
  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  return at::_convolution(input, weight, bias, stride, padding, dilation,
    transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled) {
  return at::_convolution(input, weight, bias, stride, padding, dilation,
      transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
 }

at::Tensor batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, 
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  if (check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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
  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales({false}, {false}, num_ops_id);
  if (!quantized) {
    return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto bn_x = input;
  if (pre_quantized) {
    // add quantize and dequantize for input.
    const auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    bn_x = input_q.dequantize();
  }
  auto output = at::batch_norm(bn_x, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  if (check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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
    torch_ipex::insert_or_updata_observer({input}, {output}, "max_pool2d",
                                          op_id, op_inputs, op_outputs);
    return output;
  }

  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales({false}, {false}, num_ops_id);

  if (!quantized) {
    return at::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto pool_x = input;
  if (pre_quantized) {
    // add quantize and dequantize for input.
    const auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    pool_x = input_q.dequantize();
  }
  auto output = at::max_pool2d(pool_x, kernel_size, stride, padding, dilation, ceil_mode);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size) {
  if (check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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
    torch_ipex::insert_or_updata_observer({input}, {output}, "adaptive_avg_pool2d",
                                          op_id, op_inputs, op_outputs);
    return output;
  }
  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales({false}, {false}, num_ops_id);

  if (!quantized) {
    return at::adaptive_avg_pool2d(input, output_size);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto pool_x = input;
  if (pre_quantized) {
    // add quantize and dequantize for input.
    const auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    pool_x = input_q.dequantize();
  }
  auto output = at::adaptive_avg_pool2d(pool_x, output_size);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor relu(const at::Tensor& input) {
  if (check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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

  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales({false}, {false}, num_ops_id);

  if (!quantized) {
    return at::relu(input);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto relu_x = input;
  if (pre_quantized) {
    // add quantize and dequantize for input.
    const auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    relu_x = input_q.dequantize();
  }
  auto output = at::relu(relu_x);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0,  at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

at::Tensor& relu_(at::Tensor& input) {
  // always add id, but not compute scales or insert quant and dequant.
  if (torch_ipex::check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
    auto it = tensors_flow.find(input.unsafeGetTensorImpl());
    std::vector<std::string> op_inputs, op_outputs;
    if (it == tensors_flow.end()) {
      std::string op_input = "relu_." + std::to_string(op_id) + ".input";
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
    auto& output = at::relu_(input);
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

  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  return at::relu_(input);
}

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) {
  if (torch_ipex::check_int8_calibration()) {
    auto op_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
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

  int64_t num_ops_id = torch_ipex::Int8OptConfig::fetch_and_add_ops_id();
  bool quantized = torch_ipex::get_int8_quantized_status(num_ops_id);
  std::vector<std::vector<float>> scales = torch_ipex::get_int8_scales(
    {false}, {false}, num_ops_id);
  std::vector<float> w_scales = torch_ipex::get_int8_weight_scales(num_ops_id);

  if (!quantized) {
    return at::linear(input, weight, bias);
  }
  bool pre_quantized = true, post_quantized = true;
  std::tie(pre_quantized, post_quantized) = torch_ipex::get_int8_insert_quantized_status(num_ops_id);
  auto linear_x = input;
  auto linear_w = weight;
  if (pre_quantized) {
    // add quantize and dequantize for input and weight.
    auto input_q = at::quantize_per_tensor(input, scales[0][0], 0, at::kQInt8);
    linear_x = input_q.dequantize();
    if (w_scales.size() == 1) {
      auto weight_q = at::quantize_per_tensor(weight, w_scales[0], 0, at::kQInt8);
      linear_w = weight_q.dequantize();
    } else {
      // first cache the scales for imperative path.
      if (scales_casts.find(weight.unsafeGetTensorImpl()) == scales_casts.end()) {
        const auto casted_scale = at::tensor(w_scales, at::device(at::kCPU).dtype(at::kDouble));
        scales_casts.emplace(weight.unsafeGetTensorImpl(),
            val_scales{weakref_scales(weight.getIntrusivePtr()), casted_scale});
      }
      auto it = scales_casts.find(weight.unsafeGetTensorImpl());
      auto weight_scale = std::get<1>(it->second);
      auto zero_points = at::zeros(w_scales.size(), at::dtype(at::kLong));
      auto weight_q = at::quantize_per_channel(weight, weight_scale, zero_points, 0, c10::kQInt8);
      linear_w = weight_q.dequantize();
    }
  }
  auto output = at::linear(linear_x, linear_w, bias);
  // add quantize and dequantize output.
  if (post_quantized) {
    auto output_q = at::quantize_per_tensor(output, scales[1][0], 0, at::kQInt8);
    return output_q.dequantize();
  }
  return output;
}

} // namespace autocast
} // namespace cpu
} // namespace torch_ipex
