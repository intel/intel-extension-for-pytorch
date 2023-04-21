#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MatrixRef.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <dnnl.hpp>
#include <oneDNN/oneDNN.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include <utils/Settings.h>
#include <utils/SimpleTrace.h>
#include <xetla/GRU.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace xpu::xetla;
using namespace torch::autograd;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor> gru_forward_layer(
    const Tensor& input,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& weight4,
    const Tensor& hx,
    bool reverse,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional) {
  int32_t num_bias_gate = 4;
  std::vector<int64_t> output_size = {
      input.size(0), input.size(1), hidden_size};
  auto y = at::empty(output_size, input.options());
  auto hy_ = at::empty(hx.sizes(), hx.options());

  std::vector<at::Tensor> gates_i = weight1.chunk(3, /*gates*/ 0);
  auto weight_ih = at::cat({gates_i[1], gates_i[0], gates_i[2]}, /*gates*/ 0)
                       .t()
                       .contiguous();

  std::vector<at::Tensor> gates_h = weight2.chunk(3, /*gates*/ 0);
  auto weight_hh = at::cat({gates_h[1], gates_h[0], gates_h[2]}, /*gates*/ 0)
                       .t()
                       .contiguous();

  Tensor bias;
  if (has_biases) {
    std::vector<at::Tensor> b1 = weight3.chunk(3, /*output_channels*/ 0);
    std::vector<at::Tensor> b2 = weight4.chunk(3, /*output_channels*/ 0);
    bias = at::cat(
        {b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/ 0);
  } else {
    bias = at::zeros({num_bias_gate * hidden_size}, weight1.options());
  }
  // fit Bias's dtype to which oneDNN allowed
  if (weight1.scalar_type() != ScalarType::Half)
    bias = bias.to(at::kFloat);

  auto workspace = xpu::oneDNN::gru_forward(
      input,
      hx,
      weight_ih,
      weight_hh,
      bias,
      y,
      hy_,
      reverse,
      hidden_size,
      has_biases,
      train,
      bidirectional);

  return std::make_tuple(std::move(y), std::move(hy_), std::move(workspace));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> gru_backward_layer(
    const Tensor& input,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& weight4,
    const Tensor& hx,
    const Tensor& output,
    const Tensor& hy,
    const Tensor& workspace,
    const Tensor& grad_y,
    const Tensor& grad_hy,
    bool reverse,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional) {
  int32_t num_bias_gate = 4;

  std::vector<at::Tensor> gates_i = weight1.chunk(3, /*gates*/ 0);
  auto weight_ih = at::cat({gates_i[1], gates_i[0], gates_i[2]}, /*gates*/ 0)
                       .t()
                       .contiguous();

  std::vector<at::Tensor> gates_h = weight2.chunk(3, /*gates*/ 0);
  auto weight_hh = at::cat({gates_h[1], gates_h[0], gates_h[2]}, /*gates*/ 0)
                       .t()
                       .contiguous();

  Tensor bias, grad_input, grad_hx, grad_weight_ih, grad_weight_hh, grad_bias;
  if (has_biases) {
    std::vector<at::Tensor> b1 = weight3.chunk(3, /*output_channels*/ 0);
    std::vector<at::Tensor> b2 = weight4.chunk(3, /*output_channels*/ 0);
    bias = at::cat(
        {b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/ 0);
  } else {
    bias = at::zeros({num_bias_gate * hidden_size}, weight1.options());
  }
  // fit Bias's dtype to which oneDNN allowed
  if (weight1.scalar_type() != ScalarType::Half)
    bias = bias.to(at::kFloat);

  std::tie(grad_input, grad_hx, grad_weight_ih, grad_weight_hh, grad_bias) =
      xpu::oneDNN::gru_backward(
          input,
          hx,
          weight_ih,
          weight_hh,
          bias,
          output,
          hy,
          workspace,
          grad_y,
          grad_hy,
          reverse,
          hidden_size,
          has_biases,
          train,
          bidirectional);

  std::vector<at::Tensor> grad_w_1 = grad_weight_ih.permute({0, 1, 3, 4, 2})
                                         .reshape(weight1.sizes())
                                         .chunk(3, /*gates*/ 0);
  auto grad_w_ih =
      at::cat({grad_w_1[1], grad_w_1[0], grad_w_1[2]}, /*gates*/ 0);
  std::vector<at::Tensor> grad_w_2 = grad_weight_hh.permute({0, 1, 3, 4, 2})
                                         .reshape(weight2.sizes())
                                         .chunk(3, /*gates*/ 0);
  auto grad_w_hh =
      at::cat({grad_w_2[1], grad_w_2[0], grad_w_2[2]}, /*gates*/ 0);
  std::vector<at::Tensor> grad_b =
      grad_bias.reshape(bias.sizes()).chunk(4, /*output_channels*/ 0);
  auto grad_b1 =
      at::cat({grad_b[1], grad_b[0], grad_b[2]}, /*output_channels*/ 0);
  auto grad_b2 =
      at::cat({grad_b[1], grad_b[0], grad_b[3]}, /*output_channels*/ 0);
  auto grad_hx_ = grad_hx.reshape(hx.sizes());
  grad_input.resize_as_(input);
  grad_w_ih.resize_as_(weight1);
  grad_w_hh.resize_as_(weight2);
  grad_b1.resize_as_(weight3);
  grad_b2.resize_as_(weight4);
  grad_hx.resize_as_(hx);
  return std::make_tuple(
      std::move(grad_input),
      std::move(grad_w_ih),
      std::move(grad_w_hh),
      std::move(grad_b1),
      std::move(grad_b2),
      std::move(grad_hx));
}

class GRUFunction : public Function<GRUFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable w1,
      Variable w2,
      Variable w3,
      Variable w4,
      Variable hx,
      bool reverse,
      int64_t hidden_size,
      int64_t num_layers,
      bool has_biases,
      bool train,
      bool bidirectional) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "GRU forward -> at::AtenIpexTypeXPU::GRUFunction::forward");
#endif
    variable_list saved_v = {input, w1, w2, w3, w4, hx};
    ctx->saved_data["reverse"] = reverse;
    ctx->saved_data["hidden_size"] = hidden_size;
    ctx->saved_data["num_layers"] = num_layers;
    ctx->saved_data["has_biases"] = has_biases;
    ctx->saved_data["train"] = train;
    ctx->saved_data["bidirectional"] = bidirectional;

    auto result_ = gru_forward_layer(
        input,
        w1,
        w2,
        w3,
        w4,
        hx,
        reverse,
        hidden_size,
        num_layers,
        has_biases,
        train,
        bidirectional);

    saved_v.emplace_back(std::get<0>(result_));
    saved_v.emplace_back(std::get<1>(result_));
    saved_v.emplace_back(std::get<2>(result_));
    ctx->save_for_backward(saved_v);

    variable_list result = {std::get<0>(result_), std::get<1>(result_)};
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "GRU backward -> at::AtenIpexTypeXPU::GRUFunction::backward");
#endif
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto w1 = saved[1];
    auto w2 = saved[2];
    auto w3 = saved[3];
    auto w4 = saved[4];
    auto hx = saved[5];
    auto output = saved[6];
    auto hy = saved[7];
    auto workspace = saved[8];

    Tensor grad_output = grad_outputs[0].contiguous();
    Tensor grad_hy = grad_outputs[1].contiguous();

    auto reverse = ctx->saved_data["reverse"].toBool();
    auto hidden_size = ctx->saved_data["hidden_size"].toInt();
    auto has_biases = ctx->saved_data["has_biases"].toBool();
    auto num_layers = ctx->saved_data["num_layers"].toInt();
    auto train = ctx->saved_data["train"].toBool();
    auto bidirectional = ctx->saved_data["bidirectional"].toBool();

    auto result_ = gru_backward_layer(
        input,
        w1,
        w2,
        w3,
        w4,
        hx,
        output,
        hy,
        workspace,
        grad_output,
        grad_hy,
        reverse,
        hidden_size,
        num_layers,
        has_biases,
        train,
        bidirectional);
    variable_list result = {
        std::get<0>(result_),
        std::get<1>(result_),
        std::get<2>(result_),
        std::get<3>(result_),
        std::get<4>(result_),
        std::get<5>(result_),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor()};
    return result;
  }
};

std::vector<at::Tensor> rnn_layer(
    const at::Tensor& input,
    at::TensorList weights,
    const at::Tensor& hx,
    bool reverse,
    int64_t hidden_size,
    int64_t num_layers,
    bool train,
    bool bidirectional) {
  TORCH_CHECK(weights.size() == 2 || weights.size() == 4);
  Variable input_v = input;
  Variable hx_v = hx;
  if (weights.size() == 4) {
    Variable weight_0 = weights[0];
    Variable weight_1 = weights[1];
    Variable weight_2 = weights[2];
    Variable weight_3 = weights[3];
    variable_list output = GRUFunction::apply(
        input_v,
        weight_0,
        weight_1,
        weight_2,
        weight_3,
        hx_v,
        reverse,
        hidden_size,
        num_layers,
        true,
        train,
        bidirectional);
    return {output[0], output[1]};
  } else {
    Variable weight_0 = weights[0];
    Variable weight_1 = weights[1];
    Variable weight_2 = at::zeros(weights[0].sizes(), weights[0].options());
    Variable weight_3 = at::zeros(weights[1].sizes(), weights[1].options());
    variable_list output = GRUFunction::apply(
        input_v,
        weight_0,
        weight_1,
        weight_2,
        weight_3,
        hx_v,
        reverse,
        hidden_size,
        num_layers,
        false,
        train,
        bidirectional);
    return {output[0], output[1]};
  }
}

#if defined(USE_XETLA)
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
xetla_gru_forward(
    const Tensor& input,
    const Tensor& i_weights,
    const Tensor& h_weights,
    const Tensor& i_biases,
    const Tensor& h_biases,
    const Tensor& hx,
    bool reverse,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_bias,
    bool train,
    bool bidirectional,
    float drop_prob = 0) {
  int batch_size = input.size(1);
  int sequence_size = input.size(0);
  int input_size = input.size(2);
  std::vector<int64_t> output_size = {
      input.size(0), input.size(1), hidden_size};
  at::Tensor y = at::empty(output_size, input.options());
  at::Tensor hy_ = at::empty(hx.sizes(), hx.options());
  std::vector<int64_t> workspace_shape = {
      num_layers, input.size(0), input.size(1), hidden_size};
  std::vector<int64_t> io_shape = {
      num_layers, input.size(0) + 1, input.size(1), hidden_size};
  at::Tensor reset_gate = at::empty(workspace_shape, input.options());
  at::Tensor input_gate = at::empty(workspace_shape, input.options());
  at::Tensor new_gate = at::empty(workspace_shape, input.options());
  at::Tensor hgate_2 = at::empty(workspace_shape, input.options());
  at::Tensor dropout_buffer = at::empty(workspace_shape, input.options());
  at::Tensor inter_ios = at::empty(io_shape, input.options());
  at::Tensor drop_mask;
  if (num_layers > 1) {
    drop_mask = at::empty(
        {num_layers - 1, sequence_size, batch_size, hidden_size},
        input.options().dtype(at::kFloat));
    drop_mask = drop_mask.bernoulli_(1 - drop_prob);
  }

  auto Queue = dpcppGetCurrentQueue();
  xpu::xetla::gru_forward(
      input.data_ptr(),
      hx.data_ptr(),
      i_weights.data_ptr(),
      h_weights.data_ptr(),
      i_biases.data_ptr(),
      h_biases.data_ptr(),
      hy_.data_ptr(),
      y.data_ptr(),
      num_layers <= 1 ? nullptr : drop_mask.data_ptr(),
      dropout_buffer.data_ptr(),
      inter_ios.data_ptr(),
      reset_gate.data_ptr(),
      input_gate.data_ptr(),
      new_gate.data_ptr(),
      hgate_2.data_ptr(),
      input.size(1),
      input.size(2),
      hx.size(2),
      input.size(0),
      num_layers,
      Queue);

  return std::make_tuple(
      std::move(y),
      std::move(hy_),
      std::move(reset_gate),
      std::move(input_gate),
      std::move(new_gate),
      std::move(hgate_2),
      std::move(inter_ios),
      std::move(drop_mask));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> xetla_gru_backward(
    const Tensor& input,
    const Tensor& i_weights,
    const Tensor& h_weights,
    const Tensor& i_biases,
    const Tensor& h_biases,
    const Tensor& hx,
    const Tensor& y,
    const Tensor& hy,
    const Tensor& reset_gate,
    const Tensor& input_gate,
    const Tensor& new_gate,
    const Tensor& hgate_2,
    const Tensor& inter_ios,
    const Tensor& grad_y,
    const Tensor& grad_hy,
    const Tensor& drop_mask,
    bool reverse,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_bias,
    bool train,
    bool bidirectional,
    float drop_prob = 0) {
  auto Queue = dpcppGetCurrentQueue();
  int batch_size = input.size(1);
  int input_size = input.size(2);
  int sequence_size = input.size(0);

  int gate_nums = 3;
  int bias_gate_nums = 3;
  std::vector<uint64_t> grad_shape = {
      num_layers, input.size(0), input.size(1), hidden_size};
  std::vector<uint64_t> partial_grad_shape = {
      2 * input.size(0), input.size(1), hidden_size};
  /// backwards data
  at::Tensor grad_input = at::empty(input.sizes(), input.options());
  at::Tensor grad_hx = at::empty(hx.sizes(), hx.options());

  /// backwards weight
  at::Tensor bpi0_grad = at::empty(
      {num_layers, input.size(0), input.size(1), gate_nums * hidden_size},
      input.options());
  at::Tensor bpi1_grad = at::empty(
      {num_layers, input.size(0), input.size(1), gate_nums * hidden_size},
      input.options());
  at::Tensor partial_grad = at::empty(
      {2 * input.size(0), input.size(1), hidden_size},
      input.options()); /// read & write(ping pong)

  auto grad_weight_i = at::empty(i_weights.sizes(), i_weights.options());
  auto grad_weight_h = at::empty(h_weights.sizes(), h_weights.options());
  auto grad_i_bias =
      at::empty(i_biases.sizes(), i_biases.options()).to(at::kFloat);
  auto grad_h_bias =
      at::empty(h_biases.sizes(), h_biases.options()).to(at::kFloat);
  xpu::xetla::gru_backward_data(
      grad_hy.data_ptr(),
      grad_y.data_ptr(),
      grad_input.data_ptr(),
      bpi0_grad.data_ptr(),
      bpi1_grad.data_ptr(),
      partial_grad.data_ptr(),
      grad_hx.data_ptr(),
      reset_gate.data_ptr(),
      input_gate.data_ptr(),
      new_gate.data_ptr(),
      hgate_2.data_ptr(),
      inter_ios.data_ptr(),
      i_weights.data_ptr(),
      h_weights.data_ptr(),
      num_layers <= 1 ? nullptr : drop_mask.data_ptr(),
      batch_size,
      input_size,
      hidden_size,
      sequence_size,
      num_layers,
      drop_prob,
      Queue);

  xpu::xetla::gru_backward_weight(
      bpi0_grad.data_ptr(),
      bpi1_grad.data_ptr(),
      input.data_ptr(),
      inter_ios.data_ptr(),
      grad_weight_i.data_ptr(),
      grad_weight_h.data_ptr(),
      grad_i_bias.data_ptr(),
      grad_h_bias.data_ptr(),
      batch_size,
      input_size,
      hidden_size,
      sequence_size,
      num_layers,
      Queue);

  return std::make_tuple(
      std::move(grad_input),
      std::move(grad_hx),
      std::move(grad_weight_i),
      std::move(grad_weight_h),
      std::move(grad_i_bias),
      std::move(grad_h_bias));
}

class XetlaGRUFunction : public Function<XetlaGRUFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable hx,
      bool reverse,
      int64_t hidden_size,
      int64_t num_layers,
      bool has_bias,
      bool train,
      bool bidirectional,
      double drop_prob,
      Variable i_weights,
      Variable h_weights,
      Variable i_bias,
      Variable h_bias) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "Xetla GRU forward -> at::AtenIpexTypeXPU::XetlaGRUFunction::forward");
#endif
    variable_list saved_v = {input, hx};
    ctx->saved_data["reverse"] = reverse;
    ctx->saved_data["hidden_size"] = hidden_size;
    ctx->saved_data["num_layers"] = num_layers;
    ctx->saved_data["has_biases"] = has_bias;
    ctx->saved_data["train"] = train;
    ctx->saved_data["bidirectional"] = bidirectional;
    ctx->saved_data["drop_prob"] = drop_prob;

    auto result_ = xetla_gru_forward(
        input,
        i_weights,
        h_weights,
        i_bias,
        h_bias,
        hx,
        reverse,
        hidden_size,
        num_layers,
        has_bias,
        train,
        bidirectional,
        (float)drop_prob);
    saved_v.emplace_back(std::get<0>(result_)); // y
    saved_v.emplace_back(std::get<1>(result_)); // hy_
    saved_v.emplace_back(std::get<2>(result_)); // reset_gate
    saved_v.emplace_back(std::get<3>(result_)); // input_gate
    saved_v.emplace_back(std::get<4>(result_)); // new_gate
    saved_v.emplace_back(std::get<5>(result_)); // hgate_2
    saved_v.emplace_back(std::get<6>(result_)); // inter_ios
    if (num_layers > 1)
      saved_v.emplace_back(std::get<7>(result_)); // drop_mask
    else
      saved_v.emplace_back(Tensor()); // drop_mask

    saved_v.emplace_back(i_weights);
    saved_v.emplace_back(h_weights);
    if (has_bias) {
      saved_v.emplace_back(i_bias);
      saved_v.emplace_back(h_bias);
    }
    ctx->save_for_backward(saved_v);
    variable_list result = {std::get<0>(result_), std::get<1>(result_)};
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "Xetla backward -> at::AtenIpexTypeXPU::XetlaFunction::backward");
#endif
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto hx = saved[1];
    auto output = saved[2];
    auto hy = saved[3];
    auto reset_gate = saved[4];
    auto input_gate = saved[5];
    auto new_gate = saved[6];
    auto hgate_2 = saved[7];
    auto inter_ios = saved[8];
    auto drop_mask = saved[9];

    Tensor grad_output = grad_outputs[0].contiguous();
    Tensor grad_hy = grad_outputs[1].contiguous();

    auto reverse = ctx->saved_data["reverse"].toBool();
    auto hidden_size = ctx->saved_data["hidden_size"].toInt();
    auto has_biases = ctx->saved_data["has_biases"].toBool();
    auto num_layers = ctx->saved_data["num_layers"].toInt();
    auto train = ctx->saved_data["train"].toBool();
    auto bidirectional = ctx->saved_data["bidirectional"].toBool();
    auto drop_prob = ctx->saved_data["drop_prob"].toDouble();
    auto i_weights = saved[10];
    auto h_weights = saved[11];
    auto i_biases = saved[12];
    auto h_biases = saved[13];
    if (has_biases) {
      i_biases = saved[12];
      h_biases = saved[13];
    }
    auto result_ = xetla_gru_backward(
        input,
        i_weights,
        h_weights,
        i_biases,
        h_biases,
        hx,
        output,
        hy,
        reset_gate,
        input_gate,
        new_gate,
        hgate_2,
        inter_ios,
        grad_output,
        grad_hy,
        drop_mask,
        reverse,
        hidden_size,
        num_layers,
        has_biases,
        train,
        bidirectional,
        drop_prob);
    variable_list result = {
        std::get<0>(result_), // grad_input
        std::get<1>(result_), // grad_hx
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor()};

    auto grad_w_i = std::get<2>(result_);
    auto grad_w_h = std::get<3>(result_);
    auto grad_i_bias = std::get<4>(result_);
    auto grad_h_bias = std::get<5>(result_);
    result.push_back(std::move(grad_w_i));
    result.push_back(std::move(grad_w_h));
    if (has_biases) {
      result.push_back(std::move(grad_i_bias));
      result.push_back(std::move(grad_h_bias));
    }
    return result;
  }
};

bool is_xetla_gru_available(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const ScalarType dtype) {
  // TODO: XeTLA will proive a general API to check supported platform
  if (dpcppSupportFP64()) {
    if (dtype == ScalarType::BFloat16) { // TODO: support fp16
      // More shapes could be supported by adding kernel configs manually
      if (batch_size <= 1024 && input_size <= 512 && hidden_size <= 1024) {
        return true;
      }
    }
  }
  return false;
}
#endif

std::tuple<Tensor, Tensor> gru(
    const Tensor& input_,
    const Tensor& hx_,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  //  TORCH_CHECK(!train || dropout == 0.0, "onednn_rnn doesn't support
  //  dropout");
  auto compute_eng = Settings::I().get_compute_eng();
  if (compute_eng == xpu::COMPUTE_ENG::RECOMMEND ||
      compute_eng == xpu::COMPUTE_ENG::ONEDNN) {
    auto input = input_;
    if (batch_first) {
      input = input.transpose(0, 1);
    }
    input = input.contiguous();
    auto hx = hx_.contiguous();

    int64_t hidden_size = hx.size(2);
    int64_t input_size = hx.size(2);

    at::MatrixRef<at::Tensor> weights{
        params, static_cast<size_t>(has_biases ? 4 : 2)};

    auto num_directions = bidirectional ? 2 : 1;

    auto layer_input = input;
    std::vector<at::Tensor> layer_output(num_directions);
    std::vector<at::Tensor> layer_hy(num_layers * num_directions);

    for (int64_t layer = 0; layer < num_layers; layer++) {
      for (int64_t direction = 0; direction < num_directions; direction++) {
        auto index = layer * num_directions + direction;
        auto layer_weights = weights[index];
        auto layer_hx = hx[index];
        auto reverse = (direction > 0);
        auto outputs = rnn_layer(
            layer_input,
            layer_weights,
            layer_hx,
            reverse,
            hidden_size,
            num_layers,
            train,
            bidirectional);
        layer_output[direction] = outputs[0];
        layer_hy[index] = outputs[1];
      }
      layer_input = num_directions == 1
          ? layer_output[0]
          : at::cat(layer_output, /*output_channels*/ -1);

      if (dropout != 0 && train && layer < num_layers - 1) {
        layer_input = at::dropout(layer_input, dropout, /*train=*/true);
      }
    }
    auto output = layer_input;
    auto hy = at::stack(layer_hy, 0);

    if (batch_first) {
      output = output.transpose(0, 1);
    }

    return std::make_tuple(std::move(output), std::move(hy));
  }
#if defined(USE_XETLA)
  else if (
      compute_eng == xpu::COMPUTE_ENG::XETLA &&
      is_xetla_gru_available(
          input_.size(0), input_.size(2), hx_.size(2), input_.scalar_type())) {
    int num_bias_gate = 4;
    const int BLOCK_SIZE = 32;
    auto input = input_;
    if (batch_first)
      input = input.transpose(0, 1);
    input = input.contiguous();
    auto hx = hx_.contiguous();
    int input_pad_size = input.size(-1) % BLOCK_SIZE == 0
        ? 0
        : BLOCK_SIZE - input.size(-1) % BLOCK_SIZE;
    int hidden_pad_size = hx.size(-1) % BLOCK_SIZE == 0
        ? 0
        : BLOCK_SIZE - hx.size(-1) % BLOCK_SIZE;
    if (input.size(-1) % BLOCK_SIZE != 0 || hx.size(-1) % BLOCK_SIZE != 0) {
      auto input_pad = at::zeros(
          {input.size(0), input.size(1), input_pad_size}, hx_.options());
      auto hx_pad =
          at::zeros({hx.size(0), hx.size(1), hidden_pad_size}, hx_.options());
      input = at::cat({input, input_pad}, -1).contiguous();
      hx = at::cat({hx, hx_pad}, -1).contiguous();
    }
    int64_t hidden_size = hx.size(2);
    int64_t sequence_size = input.size(0);

    at::MatrixRef<at::Tensor> weights{
        params, static_cast<size_t>(has_biases ? 4 : 2)};

    std::vector<Tensor> i_weights(num_layers, weights[0][0]);
    std::vector<Tensor> h_weights(num_layers, weights[0][0]);
    std::vector<Tensor> i_biases(num_layers, weights[0][0]);
    std::vector<Tensor> h_biases(num_layers, weights[0][0]);

    auto num_directions = bidirectional ? 2 : 1;
    if ((input_.size(-1) % BLOCK_SIZE != 0 || hx_.size(-1) % BLOCK_SIZE != 0)) {
      auto i_row_pad =
          at::zeros({hidden_pad_size, input_.size(-1)}, hx_.options());
      auto i_col_pad = at::zeros({hidden_size, input_pad_size}, hx_.options());

      auto h_row_pad =
          at::zeros({hidden_pad_size, hx_.size(-1)}, hx_.options());
      auto h_col_pad = at::zeros({hidden_size, hidden_pad_size}, hx_.options());

      auto bias_pad = at::zeros({hidden_pad_size}, hx_.options());
      for (int layer = 0; layer < num_layers; layer++) {
        auto total_weights = weights[layer];
        auto i_layered_weights = total_weights[0].chunk(3, 0);
        auto h_layered_weights = total_weights[1].chunk(3, 0);
        auto i_bias = total_weights[2].chunk(3, 0);
        auto h_bias = total_weights[3].chunk(3, 0);
        for (int j = 0; j < 3; j++) {
          if (layer == 0) {
            i_layered_weights[j] =
                at::cat({i_layered_weights[j], i_row_pad}, 0);
            i_layered_weights[j] =
                at::cat({i_layered_weights[j], i_col_pad}, 1);

            h_layered_weights[j] =
                at::cat({h_layered_weights[j], h_row_pad}, 0);
            h_layered_weights[j] =
                at::cat({h_layered_weights[j], h_col_pad}, 1);
          } else {
            i_layered_weights[j] =
                at::cat({i_layered_weights[j], h_row_pad}, 0);
            i_layered_weights[j] =
                at::cat({i_layered_weights[j], h_col_pad}, 1);

            h_layered_weights[j] =
                at::cat({h_layered_weights[j], h_row_pad}, 0);
            h_layered_weights[j] =
                at::cat({h_layered_weights[j], h_col_pad}, 1);
          }
          i_bias[j] = at::cat({i_bias[j], bias_pad}, 0);
          h_bias[j] = at::cat({h_bias[j], bias_pad}, 0);
        }
        i_weights[layer] = at::cat(
            {i_layered_weights[0], i_layered_weights[1], i_layered_weights[2]},
            0);
        h_weights[layer] = at::cat(
            {h_layered_weights[0], h_layered_weights[1], h_layered_weights[2]},
            0);
        i_biases[layer] =
            at::cat({i_bias[0], i_bias[1], i_bias[2]}, 0).to(at::kFloat);
        h_biases[layer] =
            at::cat({h_bias[0], h_bias[1], h_bias[2]}, 0).to(at::kFloat);
      }
    } else {
      for (int layer = 0; layer < num_layers; layer++) {
        auto total_weights = weights[layer];
        i_weights[layer] = total_weights[0];
        h_weights[layer] = total_weights[1];
        i_biases[layer] = total_weights[2].to(at::kFloat);
        h_biases[layer] = total_weights[3].to(at::kFloat);
      }
    }

    // concat all layer weights
    for (int layer = 0; layer < num_layers; layer++) {
      i_weights[layer] = i_weights[layer].flatten();
      h_weights[layer] = h_weights[layer].flatten();
    }
    auto i_layer_cat_weights = at::cat(i_weights, 0);
    auto h_layer_cat_weights = at::cat(h_weights, 0);
    auto i_layer_cat_biases = at::cat(i_biases, 0);
    auto h_layer_cat_biases = at::cat(h_biases, 0);

    variable_list output = XetlaGRUFunction::apply(
        input,
        hx,
        false,
        hidden_size,
        num_layers,
        has_biases,
        train,
        bidirectional,
        dropout,
        i_layer_cat_weights,
        h_layer_cat_weights,
        i_layer_cat_biases,
        h_layer_cat_biases);
    auto y = output.at(0);
    auto hy = output.at(1);
    if (input_.size(-1) % BLOCK_SIZE != 0 || hx_.size(-1) % BLOCK_SIZE != 0) {
      y = at::split(y.contiguous(), hx_.size(-1), -1)[0];
      hy = at::split(hy.contiguous(), hx_.size(-1), -1)[0];
    }
    if (batch_first) {
      y = y.transpose(0, 1);
    }

    return {std::move(y), std::move(hy)};
  }
#endif
  else {
    return at::native::gru(
        input_,
        hx_,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
