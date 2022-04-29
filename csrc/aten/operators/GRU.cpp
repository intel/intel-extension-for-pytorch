#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MatrixRef.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <core/TensorImplUtils.h>
#include <dnnl.hpp>
#include <oneDNN/oneDNN.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
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

  auto input = input_;
  if (batch_first) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();
  auto hx = hx_.contiguous();

  int64_t hidden_size = hx.size(2);

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

} // namespace AtenIpexTypeXPU
} // namespace at
