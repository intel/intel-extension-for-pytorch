#include <ATen/ATen.h>
#include <ATen/AtenIpexTypeXPU.h>
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

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace torch::autograd;

namespace at {
namespace AtenIpexTypeXPU {

class LSTMFunction : public Function<LSTMFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable hx,
      Variable cx,
      Variable weight_i,
      Variable weight_h,
      Variable bias,
      bool has_biases,
      int layer_num,
      int64_t num_layers,
      double dropout,
      bool train,
      bool bidirectional) {
    variable_list saved_v = {input, hx, cx, weight_i, weight_h, bias};
    ctx->saved_data["has_biases"] = has_biases;
    ctx->saved_data["num_layers"] = num_layers;
    ctx->saved_data["layer_num"] = layer_num;
    ctx->saved_data["dropout"] = dropout;
    ctx->saved_data["train"] = train;
    ctx->saved_data["bidirectional"] = bidirectional;

    auto result_ = xpu::oneDNN::lstm(
        input,
        hx,
        cx,
        weight_i,
        weight_h,
        bias,
        layer_num,
        num_layers,
        dropout,
        train,
        bidirectional);

    saved_v.emplace_back(std::get<0>(result_));
    saved_v.emplace_back(std::get<1>(result_));
    saved_v.emplace_back(std::get<2>(result_));
    saved_v.emplace_back(std::get<3>(result_));
    ctx->save_for_backward(saved_v);

    variable_list result = {
        std::get<0>(result_), std::get<1>(result_), std::get<2>(result_)};
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto hx = saved[1];
    auto cx = saved[2];

    auto weight_i = saved[3];
    auto weight_h = saved[4];
    auto bias = saved[5];
    auto output = saved[6];
    auto hy = saved[7];
    auto cy = saved[8];
    auto workspace = saved[9];

    Tensor grad_output = grad_outputs[0];
    Tensor grad_hy = grad_outputs[1];
    Tensor grad_cy = grad_outputs[2];

    auto has_biases = ctx->saved_data["has_biases"].toBool();
    auto num_layers = ctx->saved_data["num_layers"].toInt();
    auto dropout = ctx->saved_data["dropout"].toDouble();
    auto train = ctx->saved_data["train"].toBool();
    auto bidirectional = ctx->saved_data["bidirectional"].toBool();
    auto layer_num = ctx->saved_data["layer_num"].toInt();
    auto result_ = xpu::oneDNN::lstm_backward(
        input,
        hx,
        cx,
        output,
        hy,
        cy,
        weight_i,
        weight_h,
        bias,
        workspace,
        grad_output,
        grad_hy,
        grad_cy,
        layer_num,
        num_layers,
        dropout,
        train,
        bidirectional);

    variable_list result = {
        std::get<0>(result_),
        std::get<1>(result_),
        std::get<2>(result_),
        std::get<3>(result_),
        std::get<4>(result_),
        std::get<5>(result_)};
    for (int i = 0; i < 6; i++) {
      result.emplace_back(Variable());
    }
    return result;
  }
};

std::tuple<Tensor, Tensor, Tensor> lstm(
    const Tensor& input,
    TensorList hx,
    TensorList weights,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  Variable input_v = input;
  if (batch_first) {
    input_v = input_v.transpose(0, 1).contiguous();
  }
  auto hx_list = hx[0].unbind(0);
  auto cx_list = hx[1].unbind(0);
  std::vector<Tensor> hy_arr, cy_arr;

  int32_t num_gate = 4;
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t hidden_size = hx[0].size(-1);

  for (int32_t i = 0; i < num_layers; i++) {
    std::vector<Tensor> weight_i_arr, weight_h_arr, bias_arr;
    for (int32_t j = 0; j < num_directions; j++) {
      int32_t index = (i * num_directions + j) * (has_biases ? 4 : 2);
      weight_i_arr.push_back(weights[index].t().contiguous());
      weight_h_arr.push_back(weights[index + 1].t().contiguous());
      if (has_biases) {
        bias_arr.push_back((weights[index + 2] + weights[index + 3]));
      } else {
        bias_arr.push_back(
            at::zeros({num_gate * hidden_size}, weights[0].options()));
      }
    }
    Variable weight_i_v = at::cat(weight_i_arr, 0);
    Variable weight_h_v = at::cat(weight_h_arr, 0);
    Variable bias_v = at::cat(bias_arr, 0);

    std::vector<Tensor> hx_arr_tmp, cx_arr_tmp;
    for (int32_t j = 0; j < num_directions; j++) {
      int index = i * num_directions + j;
      auto tensor_h = at::empty_like(hx_list[0]);
      auto tensor_c = at::empty_like(cx_list[0]);
      tensor_h.copy_(hx_list[index]);
      tensor_c.copy_(cx_list[index]);
      hx_arr_tmp.push_back(tensor_h);
      cx_arr_tmp.push_back(tensor_c);
    }
    Variable hx_v = at::cat(hx_arr_tmp);
    Variable cx_v = at::cat(cx_arr_tmp);

    variable_list output = LSTMFunction::apply(
        input_v,
        hx_v,
        cx_v,
        weight_i_v,
        weight_h_v,
        bias_v,
        has_biases,
        i,
        num_layers,
        dropout,
        train,
        bidirectional);
    input_v = output[0];
    hy_arr.push_back(output[1]);
    cy_arr.push_back(output[2]);

    if (dropout != 0 && train && i < num_layers - 1) {
      input_v = at::dropout(input_v, dropout, /*train=*/true);
    }
  }
  Tensor output = input_v;
  Tensor hy, cy;
  int32_t mini_batch = input_v.size(1);
  hy = at::cat(hy_arr, 0).reshape(
      {num_layers * num_directions, mini_batch, hidden_size});
  cy = at::cat(cy_arr, 0).reshape(
      {num_layers * num_directions, mini_batch, hidden_size});
  if (batch_first) {
    output = output.transpose(0, 1).contiguous();
  }
  return std::make_tuple(output, hy, cy);
}

} // namespace AtenIpexTypeXPU
} // namespace at
