#pragma once

#include <ATen/ATen.h>

#include <oneDNN/LRUCache.h>
#include <oneDNN/Runtime.h>
#include <operators/comm/Scalar.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstm_kernel_impl(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& weight_i,
    const Tensor& weight_h,
    const Tensor& bias,
    bool has_biases,
    int layer_num,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t hidden_size = hx.size(-1);
  int32_t seq_length = input.size(0);
  int32_t mini_batch = input.size(1);
  int32_t input_size = input.size(2);
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t num_gate = 4;

  auto data_t = xpu::oneDNN::get_onednn_dtype(input);
  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  Tensor hy =
      at::empty({num_directions, mini_batch, hidden_size}, hx.options());
  Tensor cy =
      at::empty({num_directions, mini_batch, hidden_size}, cx.options());

  rnn_direction dir = bidirectional ? rnn_direction::bidirectional_concat
                                    : rnn_direction::unidirectional_left2right;
  Tensor layer_x = at::empty_like(input);
  layer_x.copy_(input);
  Tensor workspace_t;

  memory::dims src_layer_0_dims = {
      seq_length, mini_batch, input_size}; // for layer=0, tnc
  memory::dims src_layer_dims = {
      seq_length, mini_batch, hidden_size * num_directions}; // for layer>0, tnc
  memory::dims src_iter_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc hx, input hidden state
  memory::dims src_iter_c_dims = {
      1, num_directions, mini_batch, hidden_size}; // ldnc cx, input cell state
  memory::dims weights_layer_0_dims = {
      1,
      num_directions,
      input_size,
      num_gate,
      hidden_size}; // for layer=0, ldigo
  memory::dims weights_layer_dims = {
      1,
      num_directions,
      hidden_size * num_directions,
      num_gate,
      hidden_size}; // for layer>0, ldigo
  memory::dims weights_iter_dims = {
      1,
      num_directions,
      hidden_size,
      num_gate,
      hidden_size}; // ldigo weight for hidden
  memory::dims bias_dims = {1, num_directions, num_gate, hidden_size}; // ldgo
  memory::dims dst_layer_dims = {
      seq_length, mini_batch, hidden_size * num_directions}; // tnc
  memory::dims dst_iter_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc   hy, output hidden state
  memory::dims dst_iter_c_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc  cy, output cell state

  int i = layer_num;
  Tensor layer_y = at::empty({dst_layer_dims}, input.options());

  auto src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_any);
  auto weights_layer_md = memory::desc(
      {i == 0 ? weights_layer_0_dims : weights_layer_dims}, data_t, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto src_iter_c_md = memory::desc({src_iter_c_dims}, data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);
  auto dst_iter_c_md = memory::desc({dst_iter_c_dims}, data_t, format_any);

  std::shared_ptr<lstm_forward::desc> lstm_forward_desc;
  lstm_forward_desc.reset(new lstm_forward::desc(
      train ? prop_kind::forward_training : prop_kind::forward_inference,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md));

  std::shared_ptr<lstm_forward::primitive_desc> lstm_forward_pd;
  lstm_forward_pd.reset(
      new lstm_forward::primitive_desc(*lstm_forward_desc, engine));

  auto weights_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? weights_layer_0_dims : weights_layer_dims},
       data_t,
       format_ldigo},
      engine,
      weight_i.data_ptr());

  auto weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo}, engine, weight_h.data_ptr());

  auto bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, bias.data_ptr());

  auto src_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_tnc},
      engine,
      layer_x.data_ptr());

  auto src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, hx.data_ptr());

  auto src_iter_c_usr_memory = dpcpp_onednn_memory(
      {{src_iter_c_dims}, data_t, format_ldnc}, engine, cx.data_ptr());

  auto dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc}, engine, layer_y.data_ptr());

  auto dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, hy.data_ptr());

  auto dst_iter_c_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_c_dims}, data_t, format_ldnc}, engine, cy.data_ptr());

  auto expected_weights_layer_md = lstm_forward_pd->weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weights_layer_memory = memory(expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, weights_layer_memory}});
  }

  auto expected_weights_iter_md = lstm_forward_pd->weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weights_iter_memory = memory(expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, weights_iter_memory}});
  }

  auto expected_bias_md = lstm_forward_pd->bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_memory = memory(expected_bias_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bias_usr_memory, bias_memory),
        strm,
        {{DNNL_ARG_FROM, bias_usr_memory}, {DNNL_ARG_TO, bias_memory}});
  }

  auto expected_src_layer_md = lstm_forward_pd->src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_memory = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_memory, src_layer_memory),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_memory},
         {DNNL_ARG_TO, src_layer_memory}});
  }

  auto expected_src_iter_md = lstm_forward_pd->src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_memory = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_memory, src_iter_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_memory}, {DNNL_ARG_TO, src_iter_memory}});
  }

  auto expected_src_iter_c_md = lstm_forward_pd->src_iter_c_desc();
  auto src_iter_c_memory = src_iter_c_usr_memory;
  if (src_iter_c_usr_memory.get_desc() != expected_src_iter_c_md) {
    src_iter_c_memory = memory(expected_src_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_c_usr_memory, src_iter_c_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_c_usr_memory},
         {DNNL_ARG_TO, src_iter_c_memory}});
  }

  auto expected_dst_layer_md = lstm_forward_pd->dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_memory = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_memory, dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_memory},
         {DNNL_ARG_TO, dst_layer_memory}});
  }

  auto expected_dst_iter_md = lstm_forward_pd->dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_memory = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_memory, dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_memory}, {DNNL_ARG_TO, dst_iter_memory}});
  }

  auto expected_dst_iter_c_md = lstm_forward_pd->dst_iter_c_desc();
  auto dst_iter_c_memory = dst_iter_c_usr_memory;
  if (dst_iter_c_usr_memory.get_desc() != expected_dst_iter_c_md) {
    dst_iter_c_memory = memory(expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_usr_memory, dst_iter_c_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_usr_memory},
         {DNNL_ARG_TO, dst_iter_c_memory}});
  }

  std::shared_ptr<lstm_forward> lstm1_forward;
  lstm1_forward.reset(new lstm_forward(*lstm_forward_pd));
  if (train) {
    auto workspace_md = lstm_forward_pd->workspace_desc();
    workspace_t = at::zeros(workspace_md.get_size(), input.options());
    auto workspace =
        dpcpp_onednn_memory(workspace_md, engine, workspace_t.data_ptr());

    DPCPP_ONEDNN_EXEC(
        *lstm1_forward,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_memory},
         {DNNL_ARG_SRC_ITER, src_iter_memory},
         {DNNL_ARG_SRC_ITER_C, src_iter_c_memory},
         {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
         {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
         {DNNL_ARG_BIAS, bias_memory},
         {DNNL_ARG_DST_LAYER, dst_layer_memory},
         {DNNL_ARG_DST_ITER, dst_iter_memory},
         {DNNL_ARG_DST_ITER_C, dst_iter_c_memory},
         {DNNL_ARG_WORKSPACE, workspace}});
  } else {
    DPCPP_ONEDNN_EXEC(
        *lstm1_forward,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_memory},
         {DNNL_ARG_SRC_ITER, src_iter_memory},
         {DNNL_ARG_SRC_ITER_C, src_iter_c_memory},
         {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
         {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
         {DNNL_ARG_BIAS, bias_memory},
         {DNNL_ARG_DST_LAYER, dst_layer_memory},
         {DNNL_ARG_DST_ITER, dst_iter_memory},
         {DNNL_ARG_DST_ITER_C, dst_iter_c_memory}});
  }

  if (dst_layer_memory != dst_layer_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_memory, dst_layer_usr_memory),
        strm,
        {{DNNL_ARG_FROM, dst_layer_memory},
         {DNNL_ARG_TO, dst_layer_usr_memory}});
  }

  if (dst_iter_memory != dst_iter_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_memory, dst_iter_usr_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_memory}, {DNNL_ARG_TO, dst_iter_usr_memory}});
  }

  if (dst_iter_c_memory != dst_iter_c_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_memory, dst_iter_c_usr_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_memory},
         {DNNL_ARG_TO, dst_iter_c_usr_memory}});
  }
  return std::make_tuple(layer_y, hy, cy, workspace_t);
}

static inline std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
lstm_kernel_impl_bwd(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const Tensor& weight_i,
    const Tensor& weight_h,
    const Tensor& bias,
    Tensor workspace_arr,
    const Tensor& grad_output,
    const Tensor& grad_hy_ori,
    const Tensor& grad_cy_ori,
    bool has_biases,
    int layer_num,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t hidden_size = hx.size(-1);
  int32_t seq_length = input.size(0);
  int32_t mini_batch = input.size(1);
  int32_t input_size = input.size(2);
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t num_gate = 4;

  Tensor grad_hy, grad_cy;
  if (grad_hy_ori.defined()) {
    grad_hy = grad_hy_ori;
  } else {
    grad_hy = at::zeros_like(hy_);
  }
  if (grad_cy_ori.defined()) {
    grad_cy = grad_cy_ori;
  } else {
    grad_cy = at::zeros_like(cy_);
  }

  Tensor layer_x = at::empty_like(input);
  layer_x.copy_(input);

  Tensor layer_y = at::empty_like(output);
  layer_y.copy_(output);

  Tensor grad_layer_y = at::empty_like(grad_output);
  grad_layer_y.copy_(grad_output);

  auto grad_hx = at::zeros_like(hx);
  auto grad_cx = at::zeros_like(cx);
  auto grad_weight_i = at::zeros_like(weight_i);
  auto grad_weight_h = at::zeros_like(weight_h);
  auto grad_bias = at::zeros_like(bias);

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  memory::dims src_layer_0_dims = {
      seq_length, mini_batch, input_size}; // for layer=0, tnc
  memory::dims src_layer_dims = {
      seq_length, mini_batch, hidden_size * num_directions}; // for layer>0, tnc
  memory::dims src_iter_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc hx, input hidden state
  memory::dims src_iter_c_dims = {
      1, num_directions, mini_batch, hidden_size}; // ldnc cx, input cell state

  memory::dims weights_layer_0_dims = {
      1,
      num_directions,
      input_size,
      num_gate,
      hidden_size}; // for layer=0, ldigo
  memory::dims weights_layer_dims = {
      1,
      num_directions,
      hidden_size * num_directions,
      num_gate,
      hidden_size}; // for layer>0, ldigo
  memory::dims weights_iter_dims = {
      1,
      num_directions,
      hidden_size,
      num_gate,
      hidden_size}; // ldigo weight for hidden
  memory::dims bias_dims = {1, num_directions, num_gate, hidden_size}; // ldgo

  memory::dims dst_layer_dims = {
      seq_length, mini_batch, hidden_size * num_directions}; // tnc
  memory::dims dst_iter_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc   hy, output hidden state
  memory::dims dst_iter_c_dims = {
      1,
      num_directions,
      mini_batch,
      hidden_size}; // ldnc  cy, output cell state

  auto grad_src = at::zeros_like(input);
  int i = layer_num;
  auto grad_layer_x = at::zeros(
      {i == 0 ? src_layer_0_dims : src_layer_dims}, grad_output.options());

  auto src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_any);
  auto weights_layer_md = memory::desc(
      {i == 0 ? weights_layer_0_dims : weights_layer_dims}, data_t, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto src_iter_c_md = memory::desc({src_iter_c_dims}, data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);
  auto dst_iter_c_md = memory::desc({dst_iter_c_dims}, data_t, format_any);
  auto diff_src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_any);
  auto diff_src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto diff_src_iter_c_md = memory::desc({src_iter_c_dims}, data_t, format_any);
  auto diff_weights_layer_md = memory::desc(
      {i == 0 ? weights_layer_0_dims : weights_layer_dims}, data_t, format_any);
  auto diff_weights_iter_md =
      memory::desc({weights_iter_dims}, data_t, format_any);
  auto diff_bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto diff_dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto diff_dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);
  auto diff_dst_iter_c_md = memory::desc({dst_iter_c_dims}, data_t, format_any);

  rnn_direction dir = bidirectional ? rnn_direction::bidirectional_concat
                                    : rnn_direction::unidirectional_left2right;

  std::shared_ptr<lstm_forward::desc> lstm_forward_desc;
  lstm_forward_desc.reset(new lstm_forward::desc(
      prop_kind::forward_training,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md));

  std::shared_ptr<lstm_forward::primitive_desc> lstm_forward_pd;
  lstm_forward_pd.reset(
      new lstm_forward::primitive_desc(*lstm_forward_desc, engine));

  std::shared_ptr<lstm_backward::desc> lstm_backward_desc;
  lstm_backward_desc.reset(new lstm_backward::desc(
      prop_kind::backward,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md,
      diff_src_layer_md,
      diff_src_iter_md,
      diff_src_iter_c_md,
      diff_weights_layer_md,
      diff_weights_iter_md,
      diff_bias_md,
      diff_dst_layer_md,
      diff_dst_iter_md,
      diff_dst_iter_c_md));

  auto src_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_tnc},
      engine,
      layer_x.data_ptr());

  auto src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, hx.data_ptr());

  auto src_iter_c_usr_memory = dpcpp_onednn_memory(
      {{src_iter_c_dims}, data_t, format_ldnc}, engine, cx.data_ptr());

  auto dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc}, engine, layer_y.data_ptr());

  auto dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, hy_.data_ptr());

  auto dst_iter_c_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_c_dims}, data_t, format_ldnc}, engine, cy_.data_ptr());

  auto weights_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? weights_layer_0_dims : weights_layer_dims},
       data_t,
       format_ldigo},
      engine,
      weight_i.data_ptr());

  auto weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo}, engine, weight_h.data_ptr());

  auto bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, bias.data_ptr());

  auto grad_src_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_dims : src_layer_dims}, data_t, format_tnc},
      engine,
      grad_layer_x.data_ptr());

  auto grad_src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, grad_hx.data_ptr());

  auto grad_src_iter_c_usr_memory = dpcpp_onednn_memory(
      {{src_iter_c_dims}, data_t, format_ldnc}, engine, grad_cx.data_ptr());

  auto grad_dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc}, engine, grad_layer_y.data_ptr());

  auto grad_dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, grad_hy.data_ptr());

  auto grad_dst_iter_c_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_c_dims}, data_t, format_ldnc}, engine, grad_cy.data_ptr());

  auto grad_weights_layer_usr_memory = dpcpp_onednn_memory(
      {{i == 0 ? weights_layer_0_dims : weights_layer_dims},
       data_t,
       format_ldigo},
      engine,
      grad_weight_i.data_ptr());

  auto grad_weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo},
      engine,
      grad_weight_h.data_ptr());

  auto grad_bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, grad_bias.data_ptr());

  std::shared_ptr<lstm_backward::primitive_desc> lstm_backward_pd;
  lstm_backward_pd.reset(new lstm_backward::primitive_desc(
      *lstm_backward_desc, engine, *lstm_forward_pd));

  auto expected_src_layer_md = lstm_forward_pd->src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_memory = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_memory, src_layer_memory),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_memory},
         {DNNL_ARG_TO, src_layer_memory}});
  }

  auto expected_src_iter_md = lstm_forward_pd->src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_memory = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_memory, src_iter_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_memory}, {DNNL_ARG_TO, src_iter_memory}});
  }

  auto expected_src_iter_c_md = lstm_forward_pd->src_iter_c_desc();
  auto src_iter_c_memory = src_iter_c_usr_memory;
  if (src_iter_c_usr_memory.get_desc() != expected_src_iter_c_md) {
    src_iter_c_memory = memory(expected_src_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_c_usr_memory, src_iter_c_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_c_usr_memory},
         {DNNL_ARG_TO, src_iter_c_memory}});
  }

  auto expected_dst_layer_md = lstm_forward_pd->dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_memory = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_memory, dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_memory},
         {DNNL_ARG_TO, dst_layer_memory}});
  }

  auto expected_dst_iter_md = lstm_forward_pd->dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_memory = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_memory, dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_memory}, {DNNL_ARG_TO, dst_iter_memory}});
  }

  auto expected_dst_iter_c_md = lstm_forward_pd->dst_iter_c_desc();
  auto dst_iter_c_memory = dst_iter_c_usr_memory;
  if (dst_iter_c_usr_memory.get_desc() != expected_dst_iter_c_md) {
    dst_iter_c_memory = memory(expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_usr_memory, dst_iter_c_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_usr_memory},
         {DNNL_ARG_TO, dst_iter_c_memory}});
  }

  auto expected_weights_layer_md = lstm_forward_pd->weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weights_layer_memory = memory(expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, weights_layer_memory}});
  }

  auto expected_weights_iter_md = lstm_forward_pd->weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weights_iter_memory = memory(expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, weights_iter_memory}});
  }

  auto expected_bias_md = lstm_backward_pd->bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_memory = memory(expected_bias_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bias_usr_memory, bias_memory),
        strm,
        {{DNNL_ARG_FROM, bias_usr_memory}, {DNNL_ARG_TO, bias_memory}});
  }

  auto bwd_expected_weights_layer_md = lstm_backward_pd->weights_layer_desc();
  auto bwd_weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != bwd_expected_weights_layer_md) {
    bwd_weights_layer_memory = memory(bwd_expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, bwd_weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, bwd_weights_layer_memory}});
  }

  auto bwd_expected_weights_iter_md = lstm_backward_pd->weights_iter_desc();
  auto bwd_weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != bwd_expected_weights_iter_md) {
    bwd_weights_iter_memory = memory(bwd_expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, bwd_weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, bwd_weights_iter_memory}});
  }

  auto grad_expected_src_layer_md = lstm_backward_pd->diff_src_layer_desc();
  auto grad_src_layer_memory = grad_src_layer_usr_memory;
  if (grad_src_layer_usr_memory.get_desc() != grad_expected_src_layer_md) {
    grad_src_layer_memory = memory(grad_expected_src_layer_md, engine);
  }

  auto grad_expected_src_iter_md = lstm_backward_pd->diff_src_iter_desc();
  auto grad_src_iter_memory = grad_src_iter_usr_memory;
  if (grad_src_iter_usr_memory.get_desc() != grad_expected_src_iter_md) {
    grad_src_iter_memory = memory(grad_expected_src_iter_md, engine);
  }

  auto grad_expected_src_iter_c_md = lstm_backward_pd->diff_src_iter_c_desc();
  auto grad_src_iter_c_memory = grad_src_iter_c_usr_memory;
  if (grad_src_iter_c_usr_memory.get_desc() != grad_expected_src_iter_c_md) {
    grad_src_iter_c_memory = memory(grad_expected_src_iter_c_md, engine);
  }

  auto grad_expected_dst_layer_md = lstm_backward_pd->diff_dst_layer_desc();
  auto grad_dst_layer_memory = grad_dst_layer_usr_memory;
  if (grad_dst_layer_usr_memory.get_desc() != grad_expected_dst_layer_md) {
    grad_dst_layer_memory = memory(grad_expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_dst_layer_usr_memory, grad_dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, grad_dst_layer_usr_memory},
         {DNNL_ARG_TO, grad_dst_layer_memory}});
  }

  auto grad_expected_dst_iter_md = lstm_backward_pd->diff_dst_iter_desc();
  auto grad_dst_iter_memory = grad_dst_iter_usr_memory;
  if (grad_dst_iter_usr_memory.get_desc() != grad_expected_dst_iter_md) {
    grad_dst_iter_memory = memory(grad_expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_dst_iter_usr_memory, grad_dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, grad_dst_iter_usr_memory},
         {DNNL_ARG_TO, grad_dst_iter_memory}});
  }

  auto grad_expected_dst_iter_c_md = lstm_backward_pd->diff_dst_iter_c_desc();
  auto grad_dst_iter_c_memory = grad_dst_iter_c_usr_memory;
  if (grad_dst_iter_c_usr_memory.get_desc() != grad_expected_dst_iter_c_md) {
    grad_dst_iter_c_memory = memory(grad_expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_dst_iter_c_usr_memory, grad_dst_iter_c_memory),
        strm,
        {{DNNL_ARG_FROM, grad_dst_iter_c_usr_memory},
         {DNNL_ARG_TO, grad_dst_iter_c_memory}});
  }

  auto grad_expected_weights_layer_md =
      lstm_backward_pd->diff_weights_layer_desc();
  auto grad_weights_layer_memory = grad_weights_layer_usr_memory;
  if (grad_weights_layer_usr_memory.get_desc() !=
      grad_expected_weights_layer_md) {
    grad_weights_layer_memory = memory(grad_expected_weights_layer_md, engine);
  }

  auto grad_expected_weights_iter_md =
      lstm_backward_pd->diff_weights_iter_desc();
  auto grad_weights_iter_memory = grad_weights_iter_usr_memory;
  if (grad_weights_iter_usr_memory.get_desc() !=
      grad_expected_weights_iter_md) {
    grad_weights_iter_memory = memory(grad_expected_weights_iter_md, engine);
  }

  auto grad_expected_bias_md = lstm_backward_pd->diff_bias_desc();
  auto grad_bias_memory = grad_bias_usr_memory;
  if (grad_bias_usr_memory.get_desc() != grad_expected_bias_md) {
    grad_bias_memory = memory(grad_expected_bias_md, engine);
  }

  auto workspace = dpcpp_onednn_memory(
      lstm_backward_pd->workspace_desc(), engine, workspace_arr.data_ptr());

  std::shared_ptr<lstm_backward> lstm_backward_p;
  lstm_backward_p.reset(new lstm_backward(*lstm_backward_pd));

  DPCPP_ONEDNN_EXEC(
      *lstm_backward_p,
      strm,
      {{DNNL_ARG_SRC_LAYER, src_layer_memory},
       {DNNL_ARG_SRC_ITER, src_iter_memory},
       {DNNL_ARG_SRC_ITER_C, src_iter_c_memory},
       {DNNL_ARG_WEIGHTS_LAYER, bwd_weights_layer_memory},
       {DNNL_ARG_WEIGHTS_ITER, bwd_weights_iter_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST_LAYER, dst_layer_memory},
       {DNNL_ARG_DST_ITER, dst_iter_memory},
       {DNNL_ARG_DST_ITER_C, dst_iter_c_memory},
       {DNNL_ARG_DIFF_DST_LAYER, grad_dst_layer_memory},
       {DNNL_ARG_DIFF_DST_ITER, grad_dst_iter_memory},
       {DNNL_ARG_DIFF_DST_ITER_C, grad_dst_iter_c_memory},
       {DNNL_ARG_WORKSPACE, workspace},
       {DNNL_ARG_DIFF_SRC_LAYER, grad_src_layer_memory},
       {DNNL_ARG_DIFF_SRC_ITER, grad_src_iter_memory},
       {DNNL_ARG_DIFF_SRC_ITER_C, grad_src_iter_c_memory},
       {DNNL_ARG_DIFF_WEIGHTS_LAYER, grad_weights_layer_memory},
       {DNNL_ARG_DIFF_WEIGHTS_ITER, grad_weights_iter_memory},
       {DNNL_ARG_DIFF_BIAS, grad_bias_memory}});

  if (grad_src_layer_usr_memory != grad_src_layer_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_src_layer_memory, grad_src_layer_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_src_layer_memory},
         {DNNL_ARG_TO, grad_src_layer_usr_memory}});
  }
  if (grad_src_iter_usr_memory != grad_src_iter_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_src_iter_memory, grad_src_iter_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_src_iter_memory},
         {DNNL_ARG_TO, grad_src_iter_usr_memory}});
  }
  if (grad_src_iter_c_usr_memory != grad_src_iter_c_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_src_iter_c_memory, grad_src_iter_c_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_src_iter_c_memory},
         {DNNL_ARG_TO, grad_src_iter_c_usr_memory}});
  }
  if (grad_weights_layer_usr_memory != grad_weights_layer_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_weights_layer_memory, grad_weights_layer_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_weights_layer_memory},
         {DNNL_ARG_TO, grad_weights_layer_usr_memory}});
  }
  if (grad_weights_iter_usr_memory != grad_weights_iter_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_weights_iter_memory, grad_weights_iter_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_weights_iter_memory},
         {DNNL_ARG_TO, grad_weights_iter_usr_memory}});
  }
  if (grad_bias_usr_memory != grad_bias_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_bias_memory, grad_bias_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_bias_memory},
         {DNNL_ARG_TO, grad_bias_usr_memory}});
  }
  grad_src = grad_layer_x;

  return std::tuple<
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor>{
      grad_src, grad_hx, grad_cx, grad_weight_i, grad_weight_h, grad_bias};
}

} // namespace oneDNN
} // namespace xpu
