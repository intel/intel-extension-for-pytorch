#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
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
  bool train;
  at::IntArrayRef batch_sizes;
  int64_t num_gates;
  int64_t num_bias_gates;

  RNNParams(const at::Tensor& input, at::IntArrayRef batch_sizes_,
      int64_t mode_, int64_t hidden_size_, int64_t num_layers_,
      bool bidirectional, bool train_) {
    mode = static_cast<dil::rnn_kind>(mode_);
    seq_length = input.size(0);
    mini_batch = input.size(1);
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

std::vector<at::Tensor> mkldnn_rnn_layer(const at::Tensor& input, const at::Tensor& weight1, const at::Tensor& weight2,
    const at::Tensor& weight3, const at::Tensor& weight4, const at::Tensor& hx_, const at::Tensor& cx_tmp, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {

  RNNParams rnn(input, batch_sizes, mode, hidden_size, num_layers, bidirectional, train);
  auto output_size = _output_size</*is_single_direction*/true>(rnn);

  auto weight_ih = _shuffle_weight(weight1, rnn.mode);
  auto weight_hh = _shuffle_weight(weight2, rnn.mode);
  auto bias = has_biases ? _shuffle_bias(weight3, weight4, rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  // per layer input size
  int64_t input_size = input.size(2);

  at::Tensor cx_;
  if (hx_.storage().unsafeGetStorageImpl() == cx_tmp.storage().unsafeGetStorageImpl()) {
    cx_ = at::clone(cx_tmp);
  } else {
    cx_ = cx_tmp;
  }
  // TODO: should we do these reorder in DevOPs??
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(hx_, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_ih, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight_hh, true);
  // cx, cy and bias should always be fp32 in bf16 inference
  dbl::comm::reorder_to_dtype(bias, at::kFloat);
  dbl::comm::reorder_to_dtype(cx_, at::kFloat);
  auto x = dbl::comm::try_gen_dil_tensor(input, {rnn.seq_length, rnn.mini_batch, input_size}, dil::format_tag::tnc);
  auto hx = dbl::comm::try_gen_dil_tensor(hx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);

  auto w1 = dbl::comm::try_gen_dil_tensor(weight_ih, {1, 1, input_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  auto w2 = dbl::comm::try_gen_dil_tensor(weight_hh, {1, 1, rnn.hidden_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);

  auto b = dbl::comm::try_gen_dil_tensor(bias, {1, 1, rnn.num_bias_gates, rnn.hidden_size}, dil::format_tag::ldgo);

  dil::tensor y, hy;
  dil::prop_kind aprop_kind = dil::prop_kind::forward;
  auto src_type = x.get_data_type();
  if (dil::data_type::s8 == src_type || dil::data_type::u8 == src_type) {
    aprop_kind = dil::prop_kind::forward_inference;
  }
  auto _rnn_kind = static_cast<dil::rnn_kind>(rnn.mode);
  if (_rnn_kind == dil::rnn_kind::LSTM) {
    dil::tensor cy;
    auto cx = dbl::comm::try_gen_dil_tensor(cx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
    dil::lstm_forward::compute({output_size.cbegin(), output_size.cend()}, x, hx, cx, w1, w2, b, y, hy, cy, reverse, aprop_kind);
    return {dbl::comm::gen_aten_tensor_by(std::move(y)), dbl::comm::gen_aten_tensor_by(std::move(hy)).reshape(hx_.sizes()), dbl::comm::gen_aten_tensor_by(std::move(cy)).reshape(cx_.sizes())};
  } else if (_rnn_kind == dil::rnn_kind::GRU) {
    dil::lbr_gru_forward::compute({output_size.cbegin(), output_size.cend()}, x, hx, w1, w2, b, y, hy, reverse);
    return {dbl::comm::gen_aten_tensor_by(std::move(y)), dbl::comm::gen_aten_tensor_by(std::move(hy)).reshape(hx_.sizes()), at::zeros(hx_.sizes(), hx_.options())};
  } else {
    TORCH_CHECK(_rnn_kind == dil::rnn_kind::RNN_RELU || _rnn_kind == dil::rnn_kind::RNN_TANH,
                "mkldnn_rnn: unsuppored rnn mode: ", rnn.mode);
    dil::rnn_forward::compute({output_size.cbegin(), output_size.cend()}, x, hx, w1, w2, b, y, hy, rnn.mode, reverse);
    return {dbl::comm::gen_aten_tensor_by(std::move(y)), dbl::comm::gen_aten_tensor_by(std::move(hy)).reshape(hx_.sizes()), at::zeros(hx_.sizes(), hx_.options())};
  }
}

std::vector<at::Tensor> mkldnn_rnn_layer_backward(const at::Tensor& input, const at::Tensor& weight1, const at::Tensor& weight2,
    const at::Tensor& weight3, const at::Tensor& weight4, const at::Tensor& hx_, const at::Tensor& cx_tmp, const at::Tensor& output, const at::Tensor& hy_,
    const at::Tensor& cy_, const at::Tensor& grad_output, const at::Tensor& grad_hy_, const at::Tensor& grad_cy_, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {

  RNNParams rnn(input, batch_sizes, mode, hidden_size, num_layers, bidirectional, train);
  auto output_size = _output_size</*is_single_direction*/true>(rnn);

  auto weight_ih = _shuffle_weight(weight1, rnn.mode);
  auto weight_hh = _shuffle_weight(weight2, rnn.mode);
  auto bias = has_biases ? _shuffle_bias(weight3, weight4, rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  at::Tensor cx_;
  if (hx_.storage().unsafeGetStorageImpl() == cx_tmp.storage().unsafeGetStorageImpl()) {
    cx_ = at::clone(cx_tmp);
  } else {
    cx_ = cx_tmp;
  }
  // TODO: should we do these reorder in DevOPs??
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(hx_, true);
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
  auto diff_hx_ = dbl::comm::gen_aten_tensor_by(std::move(diff_hx)).reshape(hx_.sizes());
  auto diff_weight_ih = dbl::comm::gen_aten_tensor_by(std::move(diff_w1.permute({0, 1, 3, 4, 2}))).reshape(weight_ih.sizes());
  auto diff_weight_hh = dbl::comm::gen_aten_tensor_by(std::move(diff_w2.permute({0, 1, 3, 4, 2}))).reshape(weight_hh.sizes());
  auto diff_bias = dbl::comm::gen_aten_tensor_by(std::move(diff_b)).reshape(bias.sizes());
  if (_rnn_kind == dil::rnn_kind::LSTM) {
    auto diff_cx_ = dbl::comm::gen_aten_tensor_by(std::move(diff_cx)).reshape(cx_.sizes());
    return {diff_input, diff_weight_ih, diff_weight_hh, diff_bias, diff_bias, diff_hx_, diff_cx_};
  }else if (_rnn_kind == dil::rnn_kind::GRU) {
    std::vector<at::Tensor> diff_w_1 = diff_weight_ih.chunk(3, /*gates*/0);
    auto diff_w_ih = at::cat({diff_w_1[1], diff_w_1[0], diff_w_1[2]}, /*gates*/0);
    std::vector<at::Tensor> diff_w_2 = diff_weight_hh.chunk(3, /*gates*/0);
    auto diff_w_hh = at::cat({diff_w_2[1], diff_w_2[0], diff_w_2[2]}, /*gates*/0);
    std::vector<at::Tensor> diff_b = diff_bias.chunk(4, /*output_channels*/0);
    auto diff_b1 = at::cat({diff_b[1], diff_b[0], diff_b[2]}, /*output_channels*/0);
    auto diff_b2 = at::cat({diff_b[1], diff_b[0], diff_b[3]}, /*output_channels*/0);
    return {diff_input, diff_w_ih, diff_w_hh, diff_b1, diff_b2, diff_hx_, at::Tensor()};
  } else {
    return {diff_input, diff_weight_ih, diff_weight_hh, diff_bias, diff_bias, diff_hx_, at::Tensor()};
  }
}

}  // namespace rnn
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex