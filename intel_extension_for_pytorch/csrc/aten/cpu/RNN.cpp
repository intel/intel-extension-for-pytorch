#include "RNN.h"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include "WeightPack.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/autocast/autocast_verbose.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/utils/utils.h"

namespace torch_ipex {
namespace cpu {

struct RNNParams {
  ideep::rnn_kind mode;
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

  RNNParams(
      const at::Tensor& input,
      at::IntArrayRef batch_sizes_,
      int64_t mode_,
      int64_t hidden_size_,
      int64_t num_layers_,
      bool bidirectional,
      bool batch_first_,
      bool train_) {
    mode = static_cast<ideep::rnn_kind>(mode_);
    batch_first = batch_first_;
    seq_length = input.size(0);
    mini_batch = input.size(1);
    input_size = input.size(2);
    hidden_size = hidden_size_;
    num_directions = bidirectional ? 2 : 1;
    num_layers = num_layers_;
    train = train_;
    batch_sizes = batch_sizes_;
    if (mode == ideep::rnn_kind::LSTM) {
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == ideep::rnn_kind::GRU) {
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

  // mkldnn memory descriptors
  using format = ideep::format_tag;
  using desc = ideep::tensor::desc;
  using dtype = ideep::tensor::data_type;
  desc src_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{seq_length, mini_batch, _input_size}, dtype, format::tnc};
  }
  desc src_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc src_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  // logical size described as ldigo
  desc weights_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc weights_iter_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc bias_desc(dtype dtype) const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype, format::ldgo};
  }
  desc dst_layer_desc(dtype dtype) const {
    return {{seq_length, mini_batch, hidden_size}, dtype, format::tnc};
  }
  desc dst_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc dst_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
};

std::vector<int64_t> _hidden_size(const RNNParams& rnn) {
  return {rnn.num_layers * rnn.num_directions, rnn.mini_batch, rnn.hidden_size};
}

template <bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  auto output_channels = is_single_direction
      ? rnn.hidden_size
      : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU gate order is different from PyTorch's which requires gates
// shuffle (let rt,zt,nt be reset, update, new gates respectively)
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
  if (torch_ipex::cpu::is_packed(weight))
    return weight;

  auto weight_t = weight.contiguous();
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<at::Tensor> gates = weight_t.chunk(3, /*gates*/ 0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/ 0);
  }
  return weight_t;
};

at::Tensor _shuffle_bias(
    const at::Tensor& bias_ih,
    const at::Tensor& bias_hh,
    int64_t fn_mode) {
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<at::Tensor> b1 = bias_ih.chunk(3, /*output_channels*/ 0);
    std::vector<at::Tensor> b2 = bias_hh.chunk(3, /*output_channels*/ 0);
    return at::cat(
        {b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]},
        /*output_channels*/ 0);
  }
  return bias_ih + bias_hh;
};

namespace {

// Helpers for working with different hidden types.
std::tuple<at::Tensor, at::Tensor> unpack_hidden(const at::Tensor& hidden) {
  return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<at::Tensor, at::Tensor> unpack_hidden(
    const std::tuple<at::Tensor, at::Tensor>& hidden) {
  return hidden;
}

template <typename hidden_type>
hidden_type pack_hidden(const at::Tensor& hx, const at::Tensor& cx) {
  static_assert(
      std::is_same<hidden_type, void>::value,
      "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template <>
at::Tensor pack_hidden<at::Tensor>(const at::Tensor& hx, const at::Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template <>
std::tuple<at::Tensor, at::Tensor> pack_hidden<
    std::tuple<at::Tensor, at::Tensor>>(
    const at::Tensor& hx,
    const at::Tensor& cx) {
  return std::make_tuple(hx, cx);
}

} // anonymous namespace

std::vector<at::Tensor> lstm_kernel(
    const at::Tensor& input,
    const at::Tensor& w0,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& w3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
  RNNParams rnn(
      input,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      bidirectional,
      batch_first,
      train);
  auto output_size = _output_size</*is_single_direction*/ true>(rnn);
  auto output = at::empty(output_size, input.options());
  auto hy_ = at::empty(hx_.sizes(), hx_.options());
  auto cy_ = at::empty(cx_.sizes(), cx_.options());

  auto weight_ih = _shuffle_weight(w0, rnn.mode);
  auto weight_hh = _shuffle_weight(w1, rnn.mode);

  auto bias = has_biases
      ? _shuffle_bias(w2, w3, rnn.mode)
      : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  // per layer input size
  int64_t input_size = input.size(2);
  auto x = torch_ipex::cpu::itensor_view_from_dense(
      input,
      rnn.src_layer_desc(input_size, get_mkldnn_dtype(input.scalar_type())));
  auto hx = torch_ipex::cpu::itensor_view_from_dense(
      hx_, rnn.src_iter_desc(get_mkldnn_dtype(hx_.scalar_type())));
  auto cx = torch_ipex::cpu::itensor_view_from_dense(
      cx_, rnn.src_iter_c_desc(get_mkldnn_dtype(cx_.scalar_type())));
  auto b = torch_ipex::cpu::itensor_view_from_dense(
      bias, rnn.bias_desc(get_mkldnn_dtype(bias.scalar_type())));
  auto y = torch_ipex::cpu::itensor_view_from_dense(
      output, rnn.dst_layer_desc(get_mkldnn_dtype(output.scalar_type())));
  auto hy = torch_ipex::cpu::itensor_view_from_dense(
      hy_, rnn.dst_iter_desc(get_mkldnn_dtype(hy_.scalar_type())));
  auto cy = torch_ipex::cpu::itensor_view_from_dense(
      cy_, rnn.dst_iter_c_desc(get_mkldnn_dtype(cy_.scalar_type())));

  ideep::tensor w1_, w2_;
  std::tie(w1_, w2_) = torch_ipex::cpu::get_lstm_packed_weight(
      weight_ih,
      weight_hh,
      input_size,
      rnn.num_gates,
      rnn.hidden_size,
      {output_size.cbegin(), output_size.cend()},
      x,
      hx,
      cx,
      b,
      reverse,
      train);

  std::vector<at::Tensor> result;
  if (train) {
    at::Tensor workspace = at::Tensor();
    auto pd = ideep::lstm_forward_training::prepare(
        x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
    workspace = torch_ipex::cpu::empty_aten_tensor_from_desc(
        pd.workspace_desc(), input.options().dtype(at::kByte));
    ideep::tensor mkldnn_workspace;
    mkldnn_workspace.init(
        pd.workspace_desc(), workspace.template data_ptr<uint8_t>());
    ideep::lstm_forward_training::compute(
        pd, x, hx, cx, w1_, w2_, b, mkldnn_workspace, y, hy, cy, reverse);
    result.reserve(4);
    result.push_back(output);
    result.push_back(hy_);
    result.push_back(cy_);
    result.push_back(workspace);
  } else {
    ideep::lstm_forward_inference::compute(
        x, hx, cx, w1_, w2_, b, y, hy, cy, reverse);
    result.reserve(3);
    result.push_back(output);
    result.push_back(hy_);
    result.push_back(cy_);
  }
  return result;
}

std::vector<at::Tensor> ipex_lstm_layer_forward(
    const at::Tensor& input,
    const at::Tensor& w0,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& w3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cpu::ipex_lstm_layer_forward\n");
#endif
  return lstm_kernel(
      input,
      w0,
      w1,
      w2,
      w3,
      hx_,
      cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);
}

std::vector<at::Tensor> IPEXLSTMOp::_forward(
    const at::Tensor& input,
    const at::Tensor& w0,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& w3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
  at::AutoNonVariableTypeMode g;
#if defined(IPEX_DISP_OP)
  printf("IPEXLSTMOp::_forward\n");
#endif
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_lstm_layer", "")
                       .typed<decltype(ipex_lstm_layer)>();
  return op.call(
      input,
      w0,
      w1,
      w2,
      w3,
      hx_,
      cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);
}

std::vector<at::Tensor> IPEXLSTMOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& w0,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& w3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLSTMOp::forward", std::vector<c10::IValue>({}));
#endif
#if defined(IPEX_DISP_OP)
  printf("IPEXLSTMOp::forward\n");
#endif
  ctx->saved_data["reverse"] = reverse;
  ctx->saved_data["mode"] = mode;
  ctx->saved_data["hidden_size"] = hidden_size;
  ctx->saved_data["num_layers"] = num_layers;
  ctx->saved_data["has_biases"] = has_biases;
  ctx->saved_data["train"] = train;
  ctx->saved_data["bidirectional"] = bidirectional;
  ctx->saved_data["batch_first"] = batch_first;
  auto outputs = _forward(
      input,
      w0,
      w1,
      w2,
      w3,
      hx_,
      cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);

  if (train) {
    ctx->save_for_backward(
        {input,
         w0,
         w1,
         w2,
         w3,
         hx_,
         cx_,
         outputs[0],
         outputs[1],
         outputs[2],
         outputs[3]});
  }
  return outputs;
}

torch::autograd::tensor_list IPEXLSTMOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IPEXLSTMOp::backward", std::vector<c10::IValue>({}));
#endif
#if defined(IPEX_DISP_OP)
  printf("IPEXLSTMOp::backward\n");
#endif
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor w0 = saved[1];
  at::Tensor w1 = saved[2];
  at::Tensor w2 = saved[3];
  at::Tensor w3 = saved[4];
  at::Tensor hx = saved[5];
  at::Tensor cx = saved[6];
  at::Tensor output = saved[7];
  at::Tensor hy = saved[8];
  at::Tensor cy = saved[9];
  at::Tensor workspace = saved[10];
  bool reverse = ctx->saved_data["reverse"].toBool();
  int64_t mode = ctx->saved_data["mode"].toInt();
  int64_t hidden_size = ctx->saved_data["hidden_size"].toInt();
  int64_t num_layers = ctx->saved_data["num_layers"].toInt();
  bool has_biases = ctx->saved_data["has_biases"].toBool();
  bool train = ctx->saved_data["train"].toBool();
  bool bidirectional = ctx->saved_data["bidirectional"].toBool();
  bool batch_first = ctx->saved_data["batch_first"].toBool();
  at::Tensor grad_output = grad_outputs[0].contiguous();
  at::Tensor grad_hy = grad_outputs[1].contiguous();
  at::Tensor grad_cy = grad_outputs[2].contiguous();
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::ipex_lstm_layer_backward", "")
          .typed<decltype(ipex_lstm_layer_backward)>();
  std::vector<at::Tensor> grad_inputs = op.call(
      input,
      w0,
      w1,
      w2,
      w3,
      hx,
      cx,
      output,
      hy,
      cy,
      grad_output,
      grad_hy,
      grad_cy,
      reverse,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      train,
      bidirectional,
      /*batch_sizes*/ {},
      batch_first,
      workspace);
  return {
      grad_inputs[0],
      grad_inputs[1],
      grad_inputs[2],
      grad_inputs[3],
      grad_inputs[4],
      grad_inputs[5],
      grad_inputs[6],
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

std::vector<at::Tensor> ipex_lstm_layer(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::cpu::ipex_lstm_layer", std::vector<c10::IValue>({}));
#endif
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cpu::ipex_lstm_layer\n");
#endif
  if (at::GradMode::is_enabled()) {
    return IPEXLSTMOp::apply(
        input,
        weight0,
        weight1,
        weight2,
        weight3,
        hx_,
        cx_,
        reverse,
        batch_sizes,
        mode,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
        train);
  }
  return IPEXLSTMOp::_forward(
      input,
      weight0,
      weight1,
      weight2,
      weight3,
      hx_,
      cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);
}

std::vector<at::Tensor> ipex_lstm_layer_backward(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& hx_,
    const at::Tensor& cx_tmp,
    const at::Tensor& output,
    const at::Tensor& hy_,
    const at::Tensor& cy_,
    const at::Tensor& grad_output,
    const at::Tensor& grad_hy,
    const at::Tensor& grad_cy,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cpu::ipex_lstm_layer_backward\n");
#endif
  RNNParams rnn(
      input,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      bidirectional,
      batch_first,
      train);
  auto output_size = _output_size</*is_single_direction*/ true>(rnn);

  auto weight_ih = _shuffle_weight(weight0, rnn.mode);
  auto weight_hh = _shuffle_weight(weight1, rnn.mode);
  auto bias = has_biases
      ? _shuffle_bias(weight2, weight3, rnn.mode)
      : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  at::Tensor cx_;
  if (hx_.storage().unsafeGetStorageImpl() ==
      cx_tmp.storage().unsafeGetStorageImpl()) {
    cx_ = at::clone(cx_tmp);
  } else {
    cx_ = cx_tmp;
  }

  // per layer input size
  int64_t input_size = input.size(2);
  auto x = torch_ipex::cpu::itensor_view_from_dense(
      input,
      rnn.src_layer_desc(input_size, get_mkldnn_dtype(input.scalar_type())));
  auto hx = torch_ipex::cpu::itensor_view_from_dense(
      hx_, rnn.src_iter_desc(get_mkldnn_dtype(hx_.scalar_type())));
  auto cx = torch_ipex::cpu::itensor_view_from_dense(
      cx_, rnn.src_iter_c_desc(get_mkldnn_dtype(cx_.scalar_type())));
  auto w1 = torch_ipex::cpu::itensor_view_from_dense(
      weight_ih,
      rnn.weights_layer_desc(
          input_size, get_mkldnn_dtype(weight_ih.scalar_type())));
  auto w2 = torch_ipex::cpu::itensor_view_from_dense(
      weight_hh,
      rnn.weights_iter_desc(get_mkldnn_dtype(weight_hh.scalar_type())));
  auto b = torch_ipex::cpu::itensor_view_from_dense(
      bias, rnn.bias_desc(get_mkldnn_dtype(bias.scalar_type())));
  auto y = torch_ipex::cpu::itensor_view_from_dense(
      output, rnn.dst_layer_desc(get_mkldnn_dtype(output.scalar_type())));
  auto hy = torch_ipex::cpu::itensor_view_from_dense(
      hy_, rnn.dst_iter_desc(get_mkldnn_dtype(hy_.scalar_type())));
  auto cy = torch_ipex::cpu::itensor_view_from_dense(
      cy_, rnn.dst_iter_c_desc(get_mkldnn_dtype(cy_.scalar_type())));

  // Create diff_* ATen tensor and corresponding ideep tensor as fp32
  auto diff_x_ =
      at::empty(input.sizes(), input.options().dtype(at::ScalarType::Float));
  auto diff_hx_ =
      at::empty(hx_.sizes(), hx_.options().dtype(at::ScalarType::Float));
  auto diff_cx_ =
      at::empty(cx_.sizes(), cx_.options().dtype(at::ScalarType::Float));
  auto diff_w1_ = at::empty(
      weight_ih.sizes(), weight_ih.options().dtype(at::ScalarType::Float));
  auto diff_w2_ = at::empty(
      weight_hh.sizes(), weight_hh.options().dtype(at::ScalarType::Float));
  auto diff_b_ =
      at::empty(bias.sizes(), bias.options().dtype(at::ScalarType::Float));

  auto diff_x = torch_ipex::cpu::itensor_view_from_dense(
      diff_x_, rnn.src_layer_desc(input_size, ideep::tensor::data_type::f32));
  auto diff_hx = torch_ipex::cpu::itensor_view_from_dense(
      diff_hx_, rnn.src_iter_desc(ideep::tensor::data_type::f32));
  auto diff_cx = torch_ipex::cpu::itensor_view_from_dense(
      diff_cx_, rnn.src_iter_c_desc(ideep::tensor::data_type::f32));
  auto diff_w1 = torch_ipex::cpu::itensor_view_from_dense(
      diff_w1_,
      rnn.weights_layer_desc(input_size, ideep::tensor::data_type::f32));
  auto diff_w2 = torch_ipex::cpu::itensor_view_from_dense(
      diff_w2_, rnn.weights_iter_desc(ideep::tensor::data_type::f32));
  auto diff_b = torch_ipex::cpu::itensor_view_from_dense(
      diff_b_, rnn.bias_desc(ideep::tensor::data_type::f32));

  // Convert grad_y, grad_hy, grad_cy to fp32 in non-fp32 backward
  ideep::tensor diff_y, diff_hy, diff_cy;
  at::Tensor grad_y_, grad_hy_, grad_cy_;
  if (input.scalar_type() != at::ScalarType::Float) {
    grad_y_ = at::empty(
        grad_output.sizes(),
        grad_output.options().dtype(at::ScalarType::Float));
    grad_y_.copy_(grad_output);
    grad_hy_ = at::empty(
        grad_hy.sizes(), grad_hy.options().dtype(at::ScalarType::Float));
    grad_hy_.copy_(grad_hy);
    grad_cy_ = at::empty(
        grad_cy.sizes(), grad_cy.options().dtype(at::ScalarType::Float));
    grad_cy_.copy_(grad_cy);

    diff_y = torch_ipex::cpu::itensor_view_from_dense(
        grad_y_, rnn.dst_layer_desc(get_mkldnn_dtype(grad_y_.scalar_type())));
    diff_hy = torch_ipex::cpu::itensor_view_from_dense(
        grad_hy_, rnn.dst_iter_desc(get_mkldnn_dtype(grad_hy_.scalar_type())));
    diff_cy = torch_ipex::cpu::itensor_view_from_dense(
        grad_cy_, rnn.dst_iter_desc(get_mkldnn_dtype(grad_cy_.scalar_type())));
  } else {
    diff_y = torch_ipex::cpu::itensor_view_from_dense(
        grad_output, rnn.dst_layer_desc(ideep::tensor::data_type::f32));
    diff_hy = torch_ipex::cpu::itensor_view_from_dense(
        grad_hy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
    diff_cy = torch_ipex::cpu::itensor_view_from_dense(
        grad_cy, rnn.dst_iter_desc(ideep::tensor::data_type::f32));
  }

  auto forward_hint = ideep::lstm_forward_training::prepare(
      x, hx, cx, w1, w2, b, y, hy, cy, reverse);
  ideep::tensor mkldnn_workspace;
  mkldnn_workspace.init(
      forward_hint.workspace_desc(), workspace.template data_ptr<uint8_t>());
  ideep::lstm_backward::compute(
      forward_hint,
      x,
      hx,
      cx,
      w1,
      w2,
      b,
      y,
      hy,
      cy,
      diff_y,
      diff_hy,
      diff_cy,
      mkldnn_workspace,
      diff_x,
      diff_hx,
      diff_cx,
      diff_w1,
      diff_w2,
      diff_b,
      reverse);
  return {diff_x_, diff_w1_, diff_w2_, diff_b_, diff_b_, diff_hx_, diff_cx_};
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen
//   tensor,
//      the weight reorder(if any) is handed automatically inside ideep (mkldnn
//      bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded
//   input sequence. b. here break a single RNN module into { num_layers *
//   num_directions } mkldnn rnn primitives
//      for future need to cover these feature gaps.
//
// TODO: a. training with dropout
//   b. padded sequence input support
//
std::tuple<at::Tensor, at::Tensor, at::Tensor> mkldnn_rnn(
    const at::Tensor& input_,
    at::TensorList weight,
    int64_t weight_stride0,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool batch_first,
    double dropout_p,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cpu::mkldnn_rnn\n");
#endif
  TORCH_CHECK(
      batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(
        !cx_.defined(), "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

  auto input = input_;
  bool is_input_packed = batch_sizes.size() != 0;
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.contiguous();

  at::MatrixRef<at::Tensor> weights{
      weight, static_cast<size_t>(weight_stride0)};

  auto num_directions = bidirectional ? 2 : 1;
  auto layer_input = input;
  std::vector<at::Tensor> layer_output(num_directions);
  std::vector<at::Tensor> layer_hy(num_layers * num_directions);
  std::vector<at::Tensor> layer_cy(num_layers * num_directions);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      TORCH_CHECK(layer_weights.size() == 2 || layer_weights.size() == 4);
      auto layer_hx = hx[index];
      auto layer_cx = cx[index];
      auto reverse = (direction > 0);
      static auto op = torch::Dispatcher::singleton()
                           .findSchemaOrThrow("torch_ipex::ipex_lstm_layer", "")
                           .typed<decltype(ipex_lstm_layer)>();
      auto outputs = op.call(
          layer_input,
          layer_weights[0],
          layer_weights[1],
          has_biases
              ? layer_weights[2]
              : at::zeros(layer_weights[0].sizes(), layer_weights[0].options()),
          has_biases
              ? layer_weights[3]
              : at::zeros(layer_weights[1].sizes(), layer_weights[1].options()),
          layer_hx,
          layer_cx,
          reverse,
          batch_sizes,
          mode,
          hidden_size,
          num_layers,
          has_biases,
          bidirectional,
          batch_first,
          train);
      layer_output[direction] = outputs[0];
      layer_hy[index] = outputs[1];
      layer_cy[index] = outputs[2];
    }

    layer_input = num_directions == 1
        ? layer_output[0]
        : at::cat(layer_output, /*output_channels*/ -1);
    if (dropout_p != 0 && train && layer < num_layers - 1) {
      layer_input = at::dropout(layer_input, dropout_p, /*train=*/true);
    }
  }
  auto output = layer_input;
  auto hy = at::stack(layer_hy, 0);
  auto cy = at::stack(layer_cy, 0);
  if (batch_first && !is_input_packed) {
    output = output.transpose(0, 1);
  }
  return std::make_tuple(output, hy, cy);
}

template <typename hidden_type>
std::pair<at::Tensor, hidden_type> mkldnn_impl(
    const at::Tensor& input,
    const hidden_type& hidden,
    at::TensorList params,
    bool has_biases,
    ideep::rnn_kind mode,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cpu::mkldnn_impl\n");
#endif
  at::Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);
  auto mkldnn_output = mkldnn_rnn(
      input,
      params,
      has_biases ? 4 : 2,
      hx,
      cx,
      static_cast<int>(mode),
      hidden_size,
      num_layers,
      has_biases,
      batch_first,
      dropout_p,
      train,
      bidirectional,
      /*batch_sizes*/ {});
  return {
      std::get<0>(mkldnn_output),
      pack_hidden<hidden_type>(
          std::get<1>(mkldnn_output), std::get<2>(mkldnn_output))};
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
std::tuple<at::Tensor, at::Tensor, at::Tensor> ipex_lstm(
    const at::Tensor& input,
    std::vector<at::Tensor> hx,
    std::vector<at::Tensor> params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("ipex_lstm", std::vector<c10::IValue>({}));
#endif
#if defined(IPEX_DISP_OP)
  printf("ipex_lstm\n");
#endif
  auto result = cpu::mkldnn_impl(
      input,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      ideep::rnn_kind::LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
  auto output = result.first;
  auto hy = std::get<0>(result.second);
  auto cy = std::get<1>(result.second);
  return std::make_tuple(output, hy, cy);
}
} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "ipex_lstm(Tensor input, Tensor[] hx, Tensor[] params, bool "
      "has_biases, int num_layers, float dropout_p, bool train, bool "
      "bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)",
      torch_ipex::ipex_lstm);
  m.impl("ipex_lstm", c10::DispatchKey::CPU, torch_ipex::ipex_lstm);
  m.def(
      "ipex_lstm_layer(Tensor input, Tensor weight0, Tensor weight1, Tensor "
      "weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] "
      "batch_sizes, int mode, int hidden_size, int num_layers, bool "
      "has_biases, bool bidirectional, bool batch_first, bool train) -> "
      "Tensor[]");
  m.impl(
      "ipex_lstm_layer",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::ipex_lstm_layer);
  m.impl(
      "ipex_lstm_layer",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::ipex_lstm_layer_forward);
  m.def(
      "ipex_lstm_layer_backward(Tensor input, Tensor weight1, Tensor "
      "weight2, Tensor weight3, Tensor weight4, Tensor hx_, Tensor cx_tmp, "
      "Tensor output, Tensor hy_, Tensor cy_, Tensor grad_output, Tensor "
      "grad_hy, Tensor grad_cy, bool reverse, int mode, int hidden_size, int "
      "num_layers, bool has_biases, bool train, bool bidirectional, int[] "
      "batch_sizes, bool batch_first, Tensor workspace) -> Tensor[]");
  m.impl(
      "ipex_lstm_layer_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::ipex_lstm_layer_backward);
}
} // namespace

namespace torch_ipex {
namespace autocast {

std::vector<at::Tensor> ipex_lstm_layer(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_lstm_layer", "")
                       .typed<decltype(ipex_lstm_layer)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("ipex_lstm_layer");
#endif
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::autocast::ipex_lstm_layer\n");
#endif
  auto target_type = get_autocast_dtype();
  // only have bf16 support now, keep fp32 for other target_type
  bool cast_to_bfloat16 = at::kBFloat16 == target_type;
  auto casted_input =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, input) : input;
  auto casted_hx_ =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, hx_) : hx_;
  auto casted_cx_ =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, cx_) : cx_;
  auto casted_weight0 =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weight0) : weight0;
  auto casted_weight1 =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weight1) : weight1;
  auto casted_weight2 =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weight2) : weight2;
  auto casted_weight3 =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, weight3) : weight3;
  return op.call(
      casted_input,
      casted_weight0,
      casted_weight1,
      casted_weight2,
      casted_weight3,
      casted_hx_,
      casted_cx_,
      reverse,
      batch_sizes,
      mode,
      hidden_size,
      num_layers,
      has_biases,
      bidirectional,
      batch_first,
      train);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("ipex_lstm_layer", torch_ipex::autocast::ipex_lstm_layer);
}

} // namespace autocast
} // namespace torch_ipex
