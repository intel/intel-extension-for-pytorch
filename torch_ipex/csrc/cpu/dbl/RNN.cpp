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
  // for prepacked weight, return the weight tensor directly
  if (cpu::ShadeDataContext::isPackedTensor(weight)) {
      return weight;
  }

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

void prepack_lstm_weights(
  const at::Tensor& weight_ih, 
  const at::Tensor& weight_hh,
  int64_t input_size,
  int64_t num_gates,
  int64_t hidden_size,
  const dil::dims& output_sizes,
  const dil::tensor& src_layer,
  const dil::tensor& src_iter,
  const dil::tensor& src_iter_c,
  const dil::tensor& bias,
  dil::rnn_kind _rnn_kind,
  const bool reverse = false,
  dil::prop_kind aprop = dil::prop_kind::forward,
  const std::vector<float>& data_scale = dil::scale_t(),
  const std::vector<int32_t>& data_shift = {},
  const int weights_scale_mask = -1,
  const std::vector<float>& weights_scales = dil::scale_t(),
  const dil::engine& aengine = dil::engine::cpu_engine()) {
  if (cpu::ShadeDataContext::isPackedTensor(weight_ih) && cpu::ShadeDataContext::isPackedTensor(weight_hh)) {
      return;
  }

  dil::tensor w1, w2;
  dil::tensor::desc expected_weights_layer_desc, expected_weights_iter_desc;

  // TODO: add int8 for gru and rnn
  if (src_layer.get_data_type() == dil::data_type::u8 && _rnn_kind == dil::rnn_kind::LSTM) {
    auto weight_ih_ = weight_ih.reshape({1, 1, num_gates, hidden_size, input_size}).permute({0, 1, 4, 2, 3}).contiguous();
    auto weight_hh_ = weight_hh.reshape({1, 1, num_gates, hidden_size, hidden_size}).permute({0, 1, 4, 2, 3}).contiguous();

    /*
    weight_ih_ and weight_hh_ are already contiguous, so that we can set contiguous format abcde here

    We need to set the format during try gen because:
    if weight_ih if of shape {4, 1},
    after doing weight_ih.reshape(weight_ih_desc_size).permute({0, 1, 4, 2, 3}).contiguous()
    shape of weight_ih will be {1,1,1,4,1}
    stride of weight_ih will be {4,4,1,1,1}

    However, mkldnn need the stride to be {4,4,4,1,1}. Otherwise, mkldnn will consider the tensor as non-contiguous.
    As a result, we need to set the format to be abcde here when generting the dil tensor to force it to have the
    stride we need.
    */
    w1 = dbl::comm::try_gen_dil_tensor(weight_ih_, weight_ih_.sizes().vec(), dil::format_tag::abcde);
    w2 = dbl::comm::try_gen_dil_tensor(weight_hh_, weight_hh_.sizes().vec(), dil::format_tag::abcde);

    std::tie(expected_weights_layer_desc, expected_weights_iter_desc) = dil::lstm_forward::expected_weights_desc(
                        output_sizes,
                        src_layer,
                        src_iter,
                        src_iter_c,
                        w1,
                        w2,
                        bias,
                        reverse,
                        aprop,
                        data_scale,
                        data_shift,
                        weights_scale_mask,
                        weights_scales,
                        aengine);

    dil::attr_t attr;
    IPEX_CHECK(data_scale.size() == 1 && data_shift.size() == 1 && weights_scale_mask > -1 && !weights_scales.empty(), "Incorrect size for scale or zero point");
    attr.set_rnn_data_qparams(data_scale[0], data_shift[0]);
    attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);

    // attr has the scale and mask to quantize the weight
    auto expected_weight_ih = w1.reorder_if_differ_in(expected_weights_layer_desc, attr);
    auto expected_weight_hh = w2.reorder_if_differ_in(expected_weights_iter_desc, attr);

    expected_weight_ih.set_scale(weights_scales);
    expected_weight_hh.set_scale(weights_scales);

    dbl::comm::equip_dil_buffer(weight_ih, expected_weight_ih, /*padding_size*/expected_weight_ih.get_padding_size());
    dbl::comm::equip_dil_buffer(weight_hh, expected_weight_hh, /*padding_size*/expected_weight_hh.get_padding_size());
      
  } else {
    w1 = dbl::comm::try_gen_dil_tensor(weight_ih, {1, 1, input_size, num_gates, hidden_size}, dil::format_tag::ldgoi);
    w2 = dbl::comm::try_gen_dil_tensor(weight_hh, {1, 1, hidden_size, num_gates, hidden_size}, dil::format_tag::ldgoi);


    // TODO: add prepack of GRU
    if (_rnn_kind == dil::rnn_kind::LSTM) {
      std::tie(expected_weights_layer_desc, expected_weights_iter_desc) = dil::lstm_forward::expected_weights_desc(
                        output_sizes,
                        src_layer,
                        src_iter,
                        src_iter_c,
                        w1,
                        w2,
                        bias,
                        reverse,
                        aprop,
                        data_scale,
                        data_shift,
                        weights_scale_mask,
                        weights_scales,
                        aengine);
    } else {
      TORCH_CHECK(_rnn_kind == dil::rnn_kind::RNN_RELU || _rnn_kind == dil::rnn_kind::RNN_TANH,
            "mkldnn_rnn: unsuppored rnn mode: ", _rnn_kind);
      // TODO: add int8 input for rnn
      std::tie(expected_weights_layer_desc, expected_weights_iter_desc) = dil::rnn_forward::expected_weights_desc(
                          output_sizes,
                          src_layer,
                          src_iter,
                          w1,
                          w2,
                          bias,
                          _rnn_kind,
                          reverse,
                          aprop,
                          aengine);
    }

    dil::tensor expected_weight_ih {expected_weights_layer_desc};
    dil::tensor expected_weight_hh {expected_weights_iter_desc};

    expected_weight_ih.feed_from(w1);
    expected_weight_hh.feed_from(w2);

    dbl::comm::equip_dil_buffer(weight_ih, expected_weight_ih, /*padding_size*/expected_weight_ih.get_padding_size());
    dbl::comm::equip_dil_buffer(weight_hh, expected_weight_hh, /*padding_size*/expected_weight_hh.get_padding_size());
  }

  cpu::ShadeDataContext::setPackedTensor(weight_ih, true);
  cpu::ShadeDataContext::setPackedTensor(weight_hh, true);
}

std::vector<float> compute_lstm_weight_scales(const at::Tensor& weight_ih, const at::Tensor& weight_hh) {
  IPEX_CHECK(weight_ih.size(0) == weight_hh.size(0), "size(0) of weight_ih and weight_hh should equal.")
  std::vector<float> weights_scales = {};
  
  float max_s8 = static_cast<float>(std::numeric_limits<int8_t>::max());
  for (int i = 0; i < weight_ih.size(0); i++) {
    weights_scales.push_back(max_s8 / std::max(weight_ih[i].abs().max().item<float>(), weight_hh[i].abs().max().item<float>()));
  }

  return weights_scales;
}

std::vector<at::Tensor> mkldnn_rnn_layer(const at::Tensor& input, const at::Tensor& weight1, const at::Tensor& weight2,
    const at::Tensor& weight3, const at::Tensor& weight4, const at::Tensor& hx_, const at::Tensor& cx_tmp, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes, 
    const std::vector<float>& scales_from_json, const std::vector<int32_t>& shift_from_json, bool quantized) {

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

  std::vector<float> data_scale = {};
  std::vector<int32_t> data_shift = {};
  int weights_scale_mask = -1;
  std::vector<float> weights_scales = {};

  if (check_auto_mix_int8_fp32() && !check_int8_calibration() && static_cast<dil::rnn_kind>(rnn.mode) == dil::rnn_kind::LSTM) {
    if (quantized) {
      if (ShadeDataContext::isTensorMixPrecision(input, MIX_PREC_TYPE::MIX_INT8_FP32)) {
        // TODO: add check (do we fallback here if not satisfied?), should enforce the input to have scale and zero point here.
        // IPEX_CHECK(cpu::ShadeDataContext::getDilStorage(input).has_scale(), "input of u8 type should have scale");
        // IPEX_CHECK(cpu::ShadeDataContext::getDilStorage(input).has_zero_point(), "input of u8 type should have zero point");
        data_scale = cpu::ShadeDataContext::getDilStorage(input).get_scale();
        data_shift = cpu::ShadeDataContext::getDilStorage(input).get_zero_point();
      } else {
        data_scale = scales_from_json;
        data_shift = shift_from_json;
        dbl::comm::reorder_to_int8_for_mix_prec(input, data_scale, /*uint8_used*/true, data_shift);
      }

      // When feeding to mkldnn, weight is in `ldigo` but here it is in (ldgo) * i
      weights_scale_mask = 0
              + (1 << 3) // bit, indicating the unique scales for `g` dim in `ldigo`
              + (1 << 4); // bit, indicating the unique scales for `o` dim in `ldigo`

      if (ShadeDataContext::isTensorMixPrecision(weight_ih, MIX_PREC_TYPE::MIX_INT8_FP32)) {
        weights_scales = cpu::ShadeDataContext::getDilStorage(weight_ih).get_scale();
      } else {
        // If the weight has been prepacked as fp32 or bf16 dil tensor, it will be of dimension 5 of format ldigo.
        // We do not support this use case in IPEX
        if (ShadeDataContext::isDilTensor(weight_ih)) {
          IPEX_CHECK(cpu::ShadeDataContext::getDilStorage(weight_ih).ndims() == 2, "weight in int8 reference should not have been prepacked before");
        }
        weights_scales = compute_lstm_weight_scales(weight_ih, weight_hh);
      }
      dbl::comm::reorder_to_dtype(hx_, at::kFloat);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(hx_, at::kFloat);
      // TODO: check the shape after this reorder, should be (ldgo) * i
      dbl::comm::reorder_to_dtype(weight_ih, at::kFloat);
      dbl::comm::reorder_to_dtype(weight_hh, at::kFloat);
    }
  } else if (check_auto_mix_bf16_fp32()) {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
    dbl::comm::reorder_to_bf16_for_mix_prec(hx_, true);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_ih, true);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_hh, true);
  } else {
    dbl::comm::reorder_to_dtype(input, at::kFloat);
    dbl::comm::reorder_to_dtype(hx_, at::kFloat);
    dbl::comm::reorder_to_dtype(weight_ih, at::kFloat);
    dbl::comm::reorder_to_dtype(weight_hh, at::kFloat);
  }

  // cx, cy and bias should always be fp32
  dbl::comm::reorder_to_dtype(bias, at::kFloat);
  dbl::comm::reorder_to_dtype(cx_, at::kFloat);
  auto x = dbl::comm::try_gen_dil_tensor(input, {rnn.seq_length, rnn.mini_batch, input_size}, dil::format_tag::tnc);
  auto hx = dbl::comm::try_gen_dil_tensor(hx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  auto cx = dbl::comm::try_gen_dil_tensor(cx_, {1, 1, rnn.mini_batch, rnn.hidden_size}, dil::format_tag::ldnc);
  auto b = dbl::comm::try_gen_dil_tensor(bias, {1, 1, rnn.num_bias_gates, rnn.hidden_size}, dil::format_tag::ldgo);
  dil::prop_kind aprop_kind = dil::prop_kind::forward;


  auto src_type = x.get_data_type();
  if (src_type == dil::data_type::s8 || src_type ==  dil::data_type::u8) {
    aprop_kind = dil::prop_kind::forward_inference;
  }

  dil::tensor w1, w2;

  auto _rnn_kind = static_cast<dil::rnn_kind>(rnn.mode);
  // TODO: only implemented prepack for fp32 & bf16 inference of LSTM cell
  // We only do the weight prepack during the inference since 
  // the format of the weight in the FW and BW of the LSTM is different:
  // FW weight format: ldigo
  // BW weight format: ldgoi

  // TODO: prepack for GRU inference: GRU need to shuffle weight
  // We need to design how to sync the shape of the dil tensor and the aten tensor
  // during FW and BW

  // Do not prepack during int8 calibration
  if (!train && _rnn_kind != dil::rnn_kind::GRU && !check_int8_calibration()) {
    prepack_lstm_weights(
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
      _rnn_kind,
      reverse,
      aprop_kind,
      data_scale,
      data_shift,
      weights_scale_mask,
      weights_scales
    );

    w1 = dbl::comm::try_gen_dil_tensor(weight_ih);
    w2 = dbl::comm::try_gen_dil_tensor(weight_hh);
  } else {
    w1 = dbl::comm::try_gen_dil_tensor(weight_ih, {1, 1, input_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
    w2 = dbl::comm::try_gen_dil_tensor(weight_hh, {1, 1, rnn.hidden_size, rnn.num_gates, rnn.hidden_size}, dil::format_tag::ldgoi);
  }


  dil::tensor y, hy;

  if (_rnn_kind == dil::rnn_kind::LSTM) {
    dil::tensor cy;
    dil::lstm_forward::compute({output_size.cbegin(), output_size.cend()}, x, hx, cx, w1, w2, b, y, hy, cy, reverse, aprop_kind, data_scale, data_shift, weights_scale_mask, weights_scales);
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