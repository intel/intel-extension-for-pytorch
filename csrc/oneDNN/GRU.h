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

static inline void gru_forward(
    const Tensor& src_layer,
    const Tensor& src_iter,
    const Tensor& weight_layer,
    const Tensor& weight_iter,
    const Tensor& bias,
    Tensor& dst_layer,
    Tensor& dst_iter,
    Tensor& workspace,
    bool reverse,
    int64_t hidden_size,
    bool has_bias,
    bool train,
    bool bidirectional) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t seq_length = src_layer.size(0);
  int32_t mini_batch = src_layer.size(1);
  int32_t input_size = src_layer.size(2);
  int32_t num_gate = 3;
  int32_t num_bias_gate = 4;

  auto data_t = get_onednn_dtype(src_layer);
  auto bias_data_t = get_onednn_dtype(bias);
  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  rnn_direction dir = reverse ? rnn_direction::unidirectional_right2left
                              : rnn_direction::unidirectional_left2right;

  memory::dims src_layer_dims = {
      seq_length, mini_batch, input_size}; // src_layer, tnc
  memory::dims src_iter_dims = {
      1, 1, mini_batch, hidden_size}; // ldnc src_iter, src_layer hidden state
  memory::dims weights_layer_dims = {
      1, 1, input_size, num_gate, hidden_size}; // weight for src_layer, ldigo
  memory::dims weights_iter_dims = {
      1, 1, hidden_size, num_gate, hidden_size}; // weight for hidden, ldigo
  memory::dims bias_dims = {1, 1, num_bias_gate, hidden_size}; // ldgo
  memory::dims dst_layer_dims = {seq_length, mini_batch, hidden_size}; // tnc
  memory::dims dst_iter_dims = {
      1, 1, mini_batch, hidden_size}; // ldnc   dst_iter, dst_layer hidden state

  dst_layer = at::empty(dst_layer_dims, src_layer.options());
  dst_iter = at::empty(dst_iter_dims, src_iter.options());

  auto src_layer_md = memory::desc({src_layer_dims}, data_t, format_any);
  auto weights_layer_md =
      memory::desc({weights_layer_dims}, data_t, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, bias_data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);

  auto gru_forward_desc = lbr_gru_forward::desc(
      train ? prop_kind::forward_training : prop_kind::forward_inference,
      dir,
      src_layer_md,
      src_iter_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md);

  auto gru_forward_pd =
      lbr_gru_forward::primitive_desc(gru_forward_desc, engine);

  auto weights_layer_usr_memory = dpcpp_onednn_memory(
      {{weights_layer_dims}, data_t, format_ldigo},
      engine,
      weight_layer.data_ptr());

  auto weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo},
      engine,
      weight_iter.data_ptr());

  auto bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, bias.data_ptr());

  auto src_layer_usr_memory = dpcpp_onednn_memory(
      {{src_layer_dims}, data_t, format_tnc}, engine, src_layer.data_ptr());

  auto src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, src_iter.data_ptr());

  auto dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc}, engine, dst_layer.data_ptr());

  auto dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, dst_iter.data_ptr());

  auto expected_weights_layer_md = gru_forward_pd.weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weights_layer_memory = memory(expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, weights_layer_memory}});
  }

  auto expected_weights_iter_md = gru_forward_pd.weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weights_iter_memory = memory(expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, weights_iter_memory}});
  }

  auto expected_bias_md = gru_forward_pd.bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_memory = memory(expected_bias_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bias_usr_memory, bias_memory),
        strm,
        {{DNNL_ARG_FROM, bias_usr_memory}, {DNNL_ARG_TO, bias_memory}});
  }

  auto expected_src_layer_md = gru_forward_pd.src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_memory = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_memory, src_layer_memory),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_memory},
         {DNNL_ARG_TO, src_layer_memory}});
  }

  auto expected_src_iter_md = gru_forward_pd.src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_memory = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_memory, src_iter_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_memory}, {DNNL_ARG_TO, src_iter_memory}});
  }

  auto expected_dst_layer_md = gru_forward_pd.dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_memory = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_memory, dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_memory},
         {DNNL_ARG_TO, dst_layer_memory}});
  }

  auto expected_dst_iter_md = gru_forward_pd.dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_memory = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_memory, dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_memory}, {DNNL_ARG_TO, dst_iter_memory}});
  }

  auto gru_forward_p = lbr_gru_forward(gru_forward_pd);
  if (train) {
    auto workspace_md = gru_forward_pd.workspace_desc();
    auto w_size = workspace_md.get_size() / src_layer.dtype().itemsize();
    workspace = at::empty(w_size, src_layer.options());
    auto workspace_memory =
        dpcpp_onednn_memory(workspace_md, engine, workspace.data_ptr());

    DPCPP_ONEDNN_EXEC(
        gru_forward_p,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_memory},
         {DNNL_ARG_SRC_ITER, src_iter_memory},
         {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
         {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
         {DNNL_ARG_BIAS, bias_memory},
         {DNNL_ARG_DST_LAYER, dst_layer_memory},
         {DNNL_ARG_DST_ITER, dst_iter_memory},
         {DNNL_ARG_WORKSPACE, workspace_memory}});
  } else {
    DPCPP_ONEDNN_EXEC(
        gru_forward_p,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_memory},
         {DNNL_ARG_SRC_ITER, src_iter_memory},
         {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
         {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
         {DNNL_ARG_BIAS, bias_memory},
         {DNNL_ARG_DST_LAYER, dst_layer_memory},
         {DNNL_ARG_DST_ITER, dst_iter_memory}})
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
}

static inline void gru_backward(
    const Tensor& src_layer,
    const Tensor& src_iter,
    const Tensor& weight_layer,
    const Tensor& weight_iter,
    const Tensor& bias,
    const Tensor& dst_layer,
    const Tensor& dst_iter,
    const Tensor& workspace,
    const Tensor& diff_dst_layer,
    const Tensor& diff_dst_iter,
    Tensor& diff_src_layer,
    Tensor& diff_src_iter,
    Tensor& diff_weight_layer,
    Tensor& diff_weight_iter,
    Tensor& diff_bias,
    bool reverse,
    int64_t hidden_size,
    bool has_bias,
    bool train,
    bool bidirectional) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t seq_length = src_layer.size(0);
  int32_t mini_batch = src_layer.size(1);
  int32_t input_size = src_layer.size(2);
  int32_t num_gate = 3;
  int32_t num_bias_gate = 4;

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  memory::dims src_layer_dims = {
      seq_length, mini_batch, input_size}; // src_layer, tnc
  memory::dims src_iter_dims = {
      1, 1, mini_batch, hidden_size}; // ldnc src_iter, src_layer hidden state

  memory::dims weights_layer_dims = {
      1, 1, input_size, num_gate, hidden_size}; // weight for src_layer, ldigo
  memory::dims weights_iter_dims = {
      1, 1, hidden_size, num_gate, hidden_size}; // weight for hidden, ldigo
  memory::dims bias_dims = {1, 1, num_bias_gate, hidden_size}; // ldgo

  memory::dims dst_layer_dims = {seq_length, mini_batch, hidden_size}; // tnc
  memory::dims dst_iter_dims = {
      1, 1, mini_batch, hidden_size}; // ldnc   dst_iter, dst_layer hidden state

  diff_src_layer = at::empty(src_layer_dims, src_layer.options());
  diff_src_iter = at::empty(src_iter_dims, src_iter.options());
  diff_weight_layer = at::empty(weights_layer_dims, weight_layer.options());
  diff_weight_iter = at::empty(weights_iter_dims, weight_iter.options());
  diff_bias = at::empty(bias_dims, bias.options());

  auto src_layer_md = memory::desc({src_layer_dims}, data_t, format_any);
  auto weights_layer_md =
      memory::desc({weights_layer_dims}, data_t, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);
  auto diff_src_layer_md = memory::desc({src_layer_dims}, data_t, format_any);
  auto diff_src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto diff_weights_layer_md =
      memory::desc({weights_layer_dims}, data_t, format_any);
  auto diff_weights_iter_md =
      memory::desc({weights_iter_dims}, data_t, format_any);
  auto diff_bias_md = memory::desc({bias_dims}, data_t, format_any);
  auto diff_dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto diff_dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);
  rnn_direction dir = reverse ? rnn_direction::unidirectional_right2left
                              : rnn_direction::unidirectional_left2right;

  auto gru_forward_desc = lbr_gru_forward::desc(
      prop_kind::forward_training,
      dir,
      src_layer_md,
      src_iter_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md);

  auto gru_forward_pd =
      lbr_gru_forward::primitive_desc(gru_forward_desc, engine);

  auto gru_backward_desc = lbr_gru_backward::desc(
      prop_kind::backward,
      dir,
      src_layer_md,
      src_iter_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      diff_src_layer_md,
      diff_src_iter_md,
      diff_weights_layer_md,
      diff_weights_iter_md,
      diff_bias_md,
      diff_dst_layer_md,
      diff_dst_iter_md);

  auto src_layer_usr_memory = dpcpp_onednn_memory(
      {{src_layer_dims}, data_t, format_tnc}, engine, src_layer.data_ptr());

  auto src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, src_iter.data_ptr());

  auto dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc}, engine, dst_layer.data_ptr());

  auto dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, dst_iter.data_ptr());

  auto weights_layer_usr_memory = dpcpp_onednn_memory(
      {{weights_layer_dims}, data_t, format_ldigo},
      engine,
      weight_layer.data_ptr());

  auto weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo},
      engine,
      weight_iter.data_ptr());

  auto bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, bias.data_ptr());

  auto diff_src_layer_usr_memory = dpcpp_onednn_memory(
      {{src_layer_dims}, data_t, format_tnc},
      engine,
      diff_src_layer.data_ptr());

  auto diff_src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_t, format_ldnc}, engine, diff_src_iter.data_ptr());

  auto diff_dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_t, format_tnc},
      engine,
      diff_dst_layer.data_ptr());

  auto diff_dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_t, format_ldnc}, engine, diff_dst_iter.data_ptr());

  auto diff_weights_layer_usr_memory = dpcpp_onednn_memory(
      {{weights_layer_dims}, data_t, format_ldigo},
      engine,
      diff_weight_layer.data_ptr());

  auto diff_weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_t, format_ldigo},
      engine,
      diff_weight_iter.data_ptr());

  auto diff_bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_t, format_ldgo}, engine, diff_bias.data_ptr());

  auto gru_backward_pd = lbr_gru_backward::primitive_desc(
      gru_backward_desc, engine, gru_forward_pd);

  auto expected_src_layer_md = gru_forward_pd.src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_memory = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_memory, src_layer_memory),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_memory},
         {DNNL_ARG_TO, src_layer_memory}});
  }

  auto expected_src_iter_md = gru_forward_pd.src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_memory = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_memory, src_iter_memory),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_memory}, {DNNL_ARG_TO, src_iter_memory}});
  }

  auto expected_dst_layer_md = gru_forward_pd.dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_memory = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_memory, dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_memory},
         {DNNL_ARG_TO, dst_layer_memory}});
  }

  auto expected_dst_iter_md = gru_forward_pd.dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_memory = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_memory, dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_memory}, {DNNL_ARG_TO, dst_iter_memory}});
  }

  auto expected_weights_layer_md = gru_forward_pd.weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weights_layer_memory = memory(expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, weights_layer_memory}});
  }

  auto expected_weights_iter_md = gru_forward_pd.weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weights_iter_memory = memory(expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, weights_iter_memory}});
  }

  auto expected_bias_md = gru_backward_pd.bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_memory = memory(expected_bias_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bias_usr_memory, bias_memory),
        strm,
        {{DNNL_ARG_FROM, bias_usr_memory}, {DNNL_ARG_TO, bias_memory}});
  }

  auto bwd_expected_weights_layer_md = gru_backward_pd.weights_layer_desc();
  auto bwd_weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != bwd_expected_weights_layer_md) {
    bwd_weights_layer_memory = memory(bwd_expected_weights_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_layer_usr_memory, bwd_weights_layer_memory),
        strm,
        {{DNNL_ARG_FROM, weights_layer_usr_memory},
         {DNNL_ARG_TO, bwd_weights_layer_memory}});
  }

  auto bwd_expected_weights_iter_md = gru_backward_pd.weights_iter_desc();
  auto bwd_weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != bwd_expected_weights_iter_md) {
    bwd_weights_iter_memory = memory(bwd_expected_weights_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weights_iter_usr_memory, bwd_weights_iter_memory),
        strm,
        {{DNNL_ARG_FROM, weights_iter_usr_memory},
         {DNNL_ARG_TO, bwd_weights_iter_memory}});
  }

  auto expected_diff_src_layer_md = gru_backward_pd.diff_src_layer_desc();
  auto diff_src_layer_memory = diff_src_layer_usr_memory;
  if (diff_src_layer_usr_memory.get_desc() != expected_diff_src_layer_md) {
    diff_src_layer_memory = memory(expected_diff_src_layer_md, engine);
  }

  auto expected_diff_src_iter_md = gru_backward_pd.diff_src_iter_desc();
  auto diff_src_iter_memory = diff_src_iter_usr_memory;
  if (diff_src_iter_usr_memory.get_desc() != expected_diff_src_iter_md) {
    diff_src_iter_memory = memory(expected_diff_src_iter_md, engine);
  }

  auto expected_diff_dst_layer_md = gru_backward_pd.diff_dst_layer_desc();
  auto diff_dst_layer_memory = diff_dst_layer_usr_memory;
  if (diff_dst_layer_usr_memory.get_desc() != expected_diff_dst_layer_md) {
    diff_dst_layer_memory = memory(expected_diff_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_dst_layer_usr_memory, diff_dst_layer_memory),
        strm,
        {{DNNL_ARG_FROM, diff_dst_layer_usr_memory},
         {DNNL_ARG_TO, diff_dst_layer_memory}});
  }

  auto expected_diff_dst_iter_md = gru_backward_pd.diff_dst_iter_desc();
  auto diff_dst_iter_memory = diff_dst_iter_usr_memory;
  if (diff_dst_iter_usr_memory.get_desc() != expected_diff_dst_iter_md) {
    diff_dst_iter_memory = memory(expected_diff_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_dst_iter_usr_memory, diff_dst_iter_memory),
        strm,
        {{DNNL_ARG_FROM, diff_dst_iter_usr_memory},
         {DNNL_ARG_TO, diff_dst_iter_memory}});
  }

  auto expected_diff_weights_layer_md =
      gru_backward_pd.diff_weights_layer_desc();
  auto diff_weights_layer_memory = diff_weights_layer_usr_memory;
  if (diff_weights_layer_usr_memory.get_desc() !=
      expected_diff_weights_layer_md) {
    diff_weights_layer_memory = memory(expected_diff_weights_layer_md, engine);
  }

  auto expected_diff_weights_iter_md = gru_backward_pd.diff_weights_iter_desc();
  auto diff_weights_iter_memory = diff_weights_iter_usr_memory;
  if (diff_weights_iter_usr_memory.get_desc() !=
      expected_diff_weights_iter_md) {
    diff_weights_iter_memory = memory(expected_diff_weights_iter_md, engine);
  }

  auto expected_diff_bias_md = gru_backward_pd.diff_bias_desc();
  auto diff_bias_memory = diff_bias_usr_memory;
  if (diff_bias_usr_memory.get_desc() != expected_diff_bias_md) {
    diff_bias_memory = memory(expected_diff_bias_md, engine);
  }

  auto workspace_memory = dpcpp_onednn_memory(
      gru_backward_pd.workspace_desc(), engine, workspace.data_ptr());

  auto gru_backward_p = lbr_gru_backward(gru_backward_pd);

  DPCPP_ONEDNN_EXEC(
      gru_backward_p,
      strm,
      {{DNNL_ARG_SRC_LAYER, src_layer_memory},
       {DNNL_ARG_SRC_ITER, src_iter_memory},
       {DNNL_ARG_WEIGHTS_LAYER, bwd_weights_layer_memory},
       {DNNL_ARG_WEIGHTS_ITER, bwd_weights_iter_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST_LAYER, dst_layer_memory},
       {DNNL_ARG_DST_ITER, dst_iter_memory},
       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_memory},
       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_memory},
       {DNNL_ARG_WORKSPACE, workspace_memory},
       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_memory},
       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_memory},
       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_memory},
       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_memory},
       {DNNL_ARG_DIFF_BIAS, diff_bias_memory}});

  if (diff_src_layer_usr_memory != diff_src_layer_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_src_layer_memory, diff_src_layer_usr_memory),
        strm,
        {{DNNL_ARG_FROM, diff_src_layer_memory},
         {DNNL_ARG_TO, diff_src_layer_usr_memory}});
  }
  if (diff_src_iter_usr_memory != diff_src_iter_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_src_iter_memory, diff_src_iter_usr_memory),
        strm,
        {{DNNL_ARG_FROM, diff_src_iter_memory},
         {DNNL_ARG_TO, diff_src_iter_usr_memory}});
  }
  if (diff_weights_layer_usr_memory != diff_weights_layer_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_weights_layer_memory, diff_weights_layer_usr_memory),
        strm,
        {{DNNL_ARG_FROM, diff_weights_layer_memory},
         {DNNL_ARG_TO, diff_weights_layer_usr_memory}});
  }
  if (diff_weights_iter_usr_memory != diff_weights_iter_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_weights_iter_memory, diff_weights_iter_usr_memory),
        strm,
        {{DNNL_ARG_FROM, diff_weights_iter_memory},
         {DNNL_ARG_TO, diff_weights_iter_usr_memory}});
  }
  if (diff_bias_usr_memory != diff_bias_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_bias_memory, diff_bias_usr_memory),
        strm,
        {{DNNL_ARG_FROM, diff_bias_memory},
         {DNNL_ARG_TO, diff_bias_usr_memory}});
  }
}
} // namespace oneDNN
} // namespace xpu
