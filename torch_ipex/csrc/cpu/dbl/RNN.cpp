#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include "RNN.h"

#include "Common.h"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace rnn {

struct RNNParams {
  dil::rnn_kind mode;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  bool batch_first;
  bool train;
  at::IntArrayRef batch_sizes;
  int64_t num_gates;
  int64_t num_bias_gates;

  RNNParams(const at::Tensor& input, at::IntArrayRef batch_sizes_,
      int64_t mode_, int64_t hidden_size_, int64_t num_layers_,
      bool bidirectional, bool batch_first_, bool train_) {
    mode = static_cast<dil::rnn_kind>(mode_);
    batch_first = batch_first_;
    if (batch_first) {
      seq_length = input.size(1);
      mini_batch = input.size(0);
    } else {
      seq_length = input.size(0);
      mini_batch = input.size(1);
    }
    input_size = input.size(2);
    hidden_size = hidden_size_;
    num_directions = bidirectional ? 2 : 1;
    num_layers = num_layers_;
    train = train_;
    batch_sizes = batch_sizes_;
    if (mode == dil::rnn_kind::LSTM) {
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == dil::rnn_kind::GRU) {
      num_gates = 3;
      num_bias_gates = 4;
    } else {
      // RNN_RELU; RNN_TANH
      num_gates = 1;
      num_bias_gates = 1;
    }
  }

  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  // TODO: forward will not use these desc (which can only use f32 data type)
  // may need to do the same modification for backward
  // mkldnn memory descriptors
  using format = dil::format_tag;
  using desc = dil::tensor::desc;
  using dtype = dil::tensor::data_type;
  desc src_layer_desc(int64_t _input_size) const {
    return {{seq_length, mini_batch, _input_size}, dtype::f32, format::tnc};
  }
  desc src_iter_desc() const {
    return {{1, 1, mini_batch, hidden_size}, dtype::f32, format::ldnc};
  }
  // logical size described as ldigo
  desc weights_layer_desc(int64_t _input_size) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype::f32, format::ldgoi};
  }
  desc weights_iter_desc() const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype::f32, format::ldgoi};
  }
  desc bias_desc() const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype::f32, format::ldgo};
  }
  desc dst_layer_desc() const {
    return {{seq_length, mini_batch, hidden_size}, dtype::f32, format::tnc};
  }
  desc dst_iter_desc() const {
    return {{1, 1, mini_batch, hidden_size}, dtype::f32, format::ldnc};
  }
};

std::vector<int64_t> _hidden_size(const RNNParams& rnn) {
  return {rnn.num_layers * rnn.num_directions, rnn.mini_batch, rnn.hidden_size};
}

template<bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  auto output_channels = is_single_direction ? rnn.hidden_size
                                             : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU gate order is different from PyTorch's which requires gates shuffle
// (let rt,zt,nt be reset, update, new gates respectively)
//
//   MKLDNN GRU weight_ih/weight_hh gates order: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh gates order: (rt, zt, nt)
//
// MKLDNN GRU bias has 4 gates instead of 3
//  (PyTorch GRU bias)     (MKLDNN GRU bias)
//
//  bias_ih    bias_hh          bias
//  +-----+    +-----+       +---------+
//  | rt1 |    | rt2 |       | zt1+zt2 |
//  |-----|    |-----|       |---------|
//  | zt1 |    | zt2 |       | rt1+rt2 |
//  |-----|    |-----|       |---------|
//  | nt1 |    | nt2 |       |   nt1   |
//  +-----+    +-----+       |---------|
//                           |   nt2   |
//                           +---------+
//
at::Tensor _shuffle_weight(const at::Tensor& weight, int64_t fn_mode) {
  auto weight_t = weight.contiguous();
  if (static_cast<dil::rnn_kind>(fn_mode) == dil::rnn_kind::GRU) {
    std::vector<at::Tensor> gates = weight_t.chunk(3, /*gates*/0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/0);
  }
  return weight_t;
};

at::Tensor _shuffle_bias(const at::Tensor& bias_ih, const at::Tensor& bias_hh, int64_t fn_mode) {
  if (static_cast<dil::rnn_kind>(fn_mode) == dil::rnn_kind::GRU) {
    std::vector<at::Tensor> b1 = bias_ih.chunk(3, /*output_channels*/0);
    std::vector<at::Tensor> b2 = bias_hh.chunk(3, /*output_channels*/0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/0);
  }
  return bias_ih + bias_hh;
};

at::Tensor mkldnn_rnn_layer(at::Tensor& hy_, at::Tensor& cy_,
    const at::Tensor& input, at::TensorList weights,
    const at::Tensor& hx_, const at::Tensor& cx_,
    bool reverse, const RNNParams& rnn) {
  TORCH_CHECK(weights.size() == 2 || weights.size() == 4);

  auto output_size = _output_size</*is_single_direction*/true>(rnn);
  //auto output = at::empty(output_size, input.options());

  bool has_bias = weights.size() == 4;
  auto weight_ih = _shuffle_weight(weights[0], rnn.mode);
  auto weight_hh = _shuffle_weight(weights[1], rnn.mode);
  auto bias = has_bias ? _shuffle_bias(weights[2], weights[3], rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  // per layer input size
  int64_t input_size = input.size(2);
  
  // TODO: should we do these reorder in DevOPs??
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(hx_);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_ih, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_hh, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(hy_);
  // cx, cy and bias should always be fp32 in bf16 inference
  dbl::comm::reorder_to_dtype(bias, at::kFloat);
  dbl::comm::reorder_to_dtype(cx_, at::kFloat);
  dbl::comm::reorder_to_dtype(cy_, at::kFloat);
  auto x = dbl::comm::try_gen_dil_tensor(input, {rnn.seq_length, rnn.mini_batch, input_size}, dil::format_tag::tnc);
  auto hx = dbl::comm::try_gen_dil_tensor(hx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  auto w1 = dbl::comm::try_gen_dil_tensor(weight_ih, {1, 1, input_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  auto w2 = dbl::comm::try_gen_dil_tensor(weight_hh, {1, 1, rnn.hidden_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  auto b = dbl::comm::try_gen_dil_tensor(bias, {1, 1, rnn.num_bias_gates, rnn.hidden_size}, dil::format_tag::ldgo);
  auto hy = dbl::comm::try_gen_dil_tensor(hy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  //auto y = dbl::comm::try_gen_dil_tensor(output, {rnn.seq_length, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::tnc);

  dil::tensor y;
  dil::prop_kind aprop_kind = dil::prop_kind::forward;
  auto src_type = x.get_data_type();
  if (dil::data_type::s8 == src_type || dil::data_type::u8 == src_type) {
    aprop_kind = dil::prop_kind::forward_inference;
  }
  auto _rnn_kind = static_cast<dil::rnn_kind>(rnn.mode);
  if (_rnn_kind == dil::rnn_kind::LSTM) {
    auto cx = dbl::comm::try_gen_dil_tensor(cx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    auto cy = dbl::comm::try_gen_dil_tensor(cy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    dil::lstm_forward::compute({output_size.cbegin(), output_size.cend()}, x, hx, cx, w1, w2, b, y, hy, cy, reverse, aprop_kind);
  } else if (_rnn_kind == dil::rnn_kind::GRU) {
    dil::lbr_gru_forward::compute(x, hx, w1, w2, b, y, hy, reverse);
  } else {
    TORCH_CHECK(_rnn_kind == dil::rnn_kind::RNN_RELU || _rnn_kind == dil::rnn_kind::RNN_TANH,
                "mkldnn_rnn: unsuppored rnn mode: ", rnn.mode);
    dil::rnn_forward::compute(x, hx, w1, w2, b, y, hy, rnn.mode, reverse);
  }

  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mkldnn_rnn_layer_backward(const at::Tensor& input, at::TensorList weights,
    const at::Tensor& hx_, const at::Tensor& cx_,
    bool reverse, const RNNParams& rnn,
    const at::Tensor& output, const at::Tensor& hy_, const at::Tensor& cy_,
    const at::Tensor& grad_output, const at::Tensor& grad_hy_, const at::Tensor& grad_cy_) {
  TORCH_CHECK(weights.size() == 2 || weights.size() == 4);
  auto output_size = _output_size</*is_single_direction*/true>(rnn);
  bool has_bias = weights.size() == 4;
  auto weight_ih = _shuffle_weight(weights[0], rnn.mode);
  auto weight_hh = _shuffle_weight(weights[1], rnn.mode);
  auto bias = has_bias ? _shuffle_bias(weights[2], weights[3], rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  // TODO: should we do these reorder in DevOPs??
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(hx_);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_ih, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_hh, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(hy_);
  dbl::comm::reorder_to_bf16_for_mix_prec(output);
  // cx, cy and bias should always be fp32 in bf16 inference
  dbl::comm::reorder_to_dtype(bias, at::kFloat);
  dbl::comm::reorder_to_dtype(cx_, at::kFloat);
  dbl::comm::reorder_to_dtype(cy_, at::kFloat);
  dbl::comm::reorder_to_dtype(grad_output, at::kFloat);
  dbl::comm::reorder_to_dtype(grad_hy_, at::kFloat);
  dbl::comm::reorder_to_dtype(grad_cy_, at::kFloat);

  // per layer input size

  int64_t input_size = input.size(2);
  auto x = dbl::comm::try_gen_dil_tensor(input, {rnn.seq_length, rnn.mini_batch, input_size}, dil::format_tag::tnc);
  auto hx = dbl::comm::try_gen_dil_tensor(hx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  auto w1 = dbl::comm::try_gen_dil_tensor(weight_ih, {1, 1, input_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  auto w2 = dbl::comm::try_gen_dil_tensor(weight_hh, {1, 1, rnn.hidden_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  auto b = dbl::comm::try_gen_dil_tensor(bias, {1, 1, rnn.num_bias_gates, rnn.hidden_size}, dil::format_tag::ldgo);
  auto y = dbl::comm::try_gen_dil_tensor(output, {rnn.seq_length, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::tnc);
  auto hy = dbl::comm::try_gen_dil_tensor(hy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  auto diff_y = dbl::comm::try_gen_dil_tensor(grad_output, {rnn.seq_length, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::tnc);
  auto diff_hy = dbl::comm::try_gen_dil_tensor(grad_hy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  dil::tensor diff_x, diff_hx, diff_cx, diff_w1, diff_w2, diff_b;
  auto _rnn_kind = static_cast<dil::rnn_kind>(rnn.mode);
  if (_rnn_kind == dil::rnn_kind::LSTM) {
    auto cx = dbl::comm::try_gen_dil_tensor(cx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    auto cy = dbl::comm::try_gen_dil_tensor(cy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    auto diff_cy = dbl::comm::try_gen_dil_tensor(grad_cy_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    dil::lstm_backward::compute(x, hx, cx, w1, w2, b, y, hy, cy, diff_y, diff_hy, diff_cy, diff_x, diff_hx, diff_cx, diff_w1, diff_w2, diff_b, reverse);
  } else if (_rnn_kind == dil::rnn_kind::GRU) {
    dil::lbr_gru_backward::compute(x, hx, w1, w2, b, y, hy, diff_y, diff_hy, diff_x, diff_hx, diff_w1, diff_w2, diff_b, reverse);
  } else {
    TORCH_CHECK(_rnn_kind == dil::rnn_kind::RNN_RELU || _rnn_kind == dil::rnn_kind::RNN_TANH,
                "mkldnn_rnn: unsuppored rnn mode: ", rnn.mode);
    dil::rnn_backward::compute(x, hx, w1, w2, b, y, hy, diff_y, diff_hy, diff_x, diff_hx, diff_w1, diff_w2, diff_b, rnn.mode, reverse);
  }

  auto diff_input = dbl::comm::gen_aten_tensor_by(std::move(diff_x));
  auto diff_hx_ = dbl::comm::gen_aten_tensor_by(std::move(diff_hx));
  auto diff_cx_ = dbl::comm::gen_aten_tensor_by(std::move(diff_cx));
  auto diff_weight_ih = dbl::comm::gen_aten_tensor_by(std::move(diff_w1.to_dense()));
  auto diff_weight_hh = dbl::comm::gen_aten_tensor_by(std::move(diff_w2.to_dense()));
  auto diff_bias = dbl::comm::gen_aten_tensor_by(std::move(diff_b));

  return std::make_tuple(diff_input, diff_hx_, diff_cx_, diff_weight_ih, diff_weight_hh, diff_bias);
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen tensor,
//      the weight reorder(if any) is handed automatically inside dil (mkldnn bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded input sequence.
//   b. here break a single RNN module into { num_layers * num_directions } mkldnn rnn primitives
//      for future need to cover these feature gaps.
//
//TODO: a. training with dropout
//   b. padded sequence input support
//
std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>> mkldnn_rnn(
    const at::Tensor& input_, std::vector<at::Tensor> weight, int64_t weight_stride0,
    const at::Tensor& hx_, const at::Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, at::IntArrayRef batch_sizes) {
  TORCH_CHECK(!train || dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<dil::rnn_kind>(mode) != dil::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

  RNNParams fn(input_, batch_sizes, mode, hidden_size, num_layers, bidirectional, batch_first, train);

  auto input = input_;
  if (batch_first && !fn.is_input_packed()) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.defined() ? cx_.contiguous() : at::Tensor();

  auto hy = at::empty(_hidden_size(fn), hx.options());
  // NB: Not allowed to return undefined tensors
  auto cy = cx.defined() ? at::empty(_hidden_size(fn), cx.options())
                         : at::empty({0}, hx.options());

  at::MatrixRef<at::Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_directions = fn.num_directions;
  auto layer_input = input;
  std::vector<at::Tensor> layer_output(num_layers * num_directions);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : at::Tensor();
      auto layer_cy = cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto reverse = (direction > 0);
      layer_output[index] = mkldnn_rnn_layer(layer_hy, layer_cy, layer_input, layer_weights, layer_hx, layer_cx, reverse, fn);
    }
    layer_input = num_directions == 1 ? layer_output[layer * num_directions]
                                      : at::cat({layer_output[layer * num_directions], layer_output[layer * num_directions + 1]}, /*output_channels*/-1);
  }
  auto output = layer_input;

  if (batch_first && !fn.is_input_packed()) {
    output = output.transpose(0, 1);
  }

  return std::make_tuple(output, hy, cy, layer_output);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mkldnn_rnn_backward(
    const at::Tensor& input_, std::vector<at::Tensor> weight, int64_t weight_stride0,
    const at::Tensor& hx_, const at::Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, at::IntArrayRef batch_sizes,
    std::vector<at::Tensor> outputs, const at::Tensor& grad_output_, const at::Tensor& grad_hy, const at::Tensor& grad_cy, std::vector<at::Tensor> layer_outputs) {
  TORCH_CHECK(!train || dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<dil::rnn_kind>(mode) != dil::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

  RNNParams fn(input_, batch_sizes, mode, hidden_size, num_layers, bidirectional, batch_first, train);
  auto num_directions = fn.num_directions;

  auto input = input_;
  auto grad_output = grad_output_;
  if (batch_first && !fn.is_input_packed()) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.defined() ? cx_.contiguous() : at::Tensor();
  auto dhy = grad_hy.contiguous();
  auto dcy = grad_cy.contiguous();
  at::MatrixRef<at::Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto output = outputs[0].contiguous();
  auto hy = outputs[1].contiguous();
  auto cy = outputs[2].contiguous();
  at::Tensor grad_input, grad_hx, grad_cx, grad_w1, grad_w2, grad_b;
  std::vector<at::Tensor> layer_grad_input(num_directions);
  std::vector<at::Tensor> layer_grad_hx(num_directions);
  std::vector<at::Tensor> layer_grad_cx(num_directions);
  std::vector<at::Tensor> layer_grad_w1(num_directions);
  std::vector<at::Tensor> layer_grad_w2(num_directions);
  std::vector<at::Tensor> layer_grad_b(num_directions);
  std::vector<at::Tensor> layer_grad_outputs(num_directions);
  layer_grad_outputs = num_directions == 1 ? std::vector<at::Tensor>{grad_output} : at::chunk(grad_output, 2, -1);
  for (int64_t layer = num_layers - 1; layer >= 0; layer--) {
    auto layer_input = layer == 0 ? input : (num_directions == 1 ? layer_outputs[(layer - 1) * num_directions] : at::cat({layer_outputs[(layer - 1) * num_directions].contiguous(), layer_outputs[(layer - 1) * num_directions + 1].contiguous()}, /*output_channels*/-1));
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : at::Tensor();
      auto layer_cy = cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto reverse = (direction > 0);
      auto layer_grad_hy = dhy[index];
      auto layer_grad_cy = dcy[index];
      auto layer_output = layer_outputs[index];
      std::tie(layer_grad_input[direction], layer_grad_hx[direction], layer_grad_cx[direction], layer_grad_w1[direction], layer_grad_w2[direction], layer_grad_b[direction]) = mkldnn_rnn_layer_backward(layer_input, layer_weights, layer_hx, layer_cx, reverse, fn, layer_output.contiguous(), layer_hy, layer_cy, layer_grad_outputs[direction].contiguous(), layer_grad_hy, layer_grad_cy);
    }
    grad_input = num_directions == 1? layer_grad_input[0] : layer_grad_input[0] + layer_grad_input[1];
    layer_grad_outputs = num_directions == 1 ? std::vector<at::Tensor>{grad_input} : at::chunk(grad_input, 2, -1);
  }

  if (batch_first && !fn.is_input_packed()) {
    grad_input = grad_input.transpose(0, 1);
  }

  return std::make_tuple(grad_input, grad_hx, grad_cx, grad_w1, grad_w2, grad_b);
}

}  // namespace rnn
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex