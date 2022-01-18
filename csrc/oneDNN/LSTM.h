#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

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

static inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstm(
    const Tensor& src,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& wgh_i,
    const Tensor& wgh_h,
    const Tensor& bia,
    int layer_num,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t hidden_sz = hx.size(-1);
  int32_t seq_length = src.size(0);
  int32_t mini_batch = src.size(1);
  int32_t src_sz = src.size(2);
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t num_gate = 4;

  auto src_data_t = xpu::oneDNN::get_onednn_dtype(src);
  auto iter_c_data_t = xpu::oneDNN::get_onednn_dtype(src);

  auto bia_ = bia;
  auto cx_ = cx;
  auto bia_data_t = xpu::oneDNN::get_onednn_dtype(bia_);
  if (bia_.scalar_type() == ScalarType::BFloat16) {
    bia_ = bia_.to(at::kFloat);
    cx_ = cx_.to(at::kFloat);
    bia_data_t = memory::data_type::f32;
    iter_c_data_t = memory::data_type::f32;
  }

  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  Tensor hy = at::empty({num_directions, mini_batch, hidden_sz}, hx.options());
  Tensor cy = at::empty({num_directions, mini_batch, hidden_sz}, cx_.options());

  rnn_direction dir = bidirectional ? rnn_direction::bidirectional_concat
                                    : rnn_direction::unidirectional_left2right;
  Tensor layer_x = at::empty_like(src);
  layer_x.copy_(src);
  Tensor workspace_t;

  memory::dims src_layer_0_tz = {
      seq_length, mini_batch, src_sz}; // for layer=0, tnc
  memory::dims src_layer_tz = {
      seq_length, mini_batch, hidden_sz * num_directions}; // for layer>0, tnc
  memory::dims src_iter_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc hx, src hidden state
  memory::dims src_iter_c_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc cx_, src cell state
  memory::dims wghs_layer_0_tz = {
      1, num_directions, src_sz, num_gate, hidden_sz}; // for layer=0, ldigo
  memory::dims wghs_layer_tz = {
      1,
      num_directions,
      hidden_sz * num_directions,
      num_gate,
      hidden_sz}; // for layer>0, ldigo
  memory::dims wghs_iter_tz = {
      1,
      num_directions,
      hidden_sz,
      num_gate,
      hidden_sz}; // ldigo wgh for hidden
  memory::dims bia_tz = {1, num_directions, num_gate, hidden_sz}; // ldgo
  memory::dims dst_layer_tz = {
      seq_length, mini_batch, hidden_sz * num_directions}; // tnc
  memory::dims dst_iter_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc   hy, dst hidden state
  memory::dims dst_iter_c_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc  cy, dst cell state

  int i = layer_num;
  Tensor layer_y = at::empty({dst_layer_tz}, src.options());

  auto src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_tz : src_layer_tz}, src_data_t, format_any);
  auto wghs_layer_md = memory::desc(
      {i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, src_data_t, format_any);
  auto wghs_iter_md = memory::desc({wghs_iter_tz}, src_data_t, format_any);
  auto bia_md = memory::desc({bia_tz}, bia_data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_tz}, src_data_t, format_any);
  auto src_iter_c_md = memory::desc({src_iter_c_tz}, iter_c_data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_tz}, src_data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_tz}, src_data_t, format_any);
  auto dst_iter_c_md = memory::desc({dst_iter_c_tz}, iter_c_data_t, format_any);

  std::shared_ptr<lstm_forward::desc> lstm_forward_desc;
  lstm_forward_desc.reset(new lstm_forward::desc(
      train ? prop_kind::forward_training : prop_kind::forward_inference,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      wghs_layer_md,
      wghs_iter_md,
      bia_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md));

  std::shared_ptr<lstm_forward::primitive_desc> lstm_forward_pd;
  lstm_forward_pd.reset(
      new lstm_forward::primitive_desc(*lstm_forward_desc, engine));

  auto wghs_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, src_data_t, format_ldigo},
      engine,
      wgh_i.data_ptr());

  auto wghs_iter_usr_m = dpcpp_onednn_memory(
      {{wghs_iter_tz}, src_data_t, format_ldigo}, engine, wgh_h.data_ptr());

  auto bia_usr_m = dpcpp_onednn_memory(
      {{bia_tz}, bia_data_t, format_ldgo}, engine, bia_.data_ptr());

  auto src_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_tz : src_layer_tz}, src_data_t, format_tnc},
      engine,
      layer_x.data_ptr());

  auto src_iter_usr_m = dpcpp_onednn_memory(
      {{src_iter_tz}, src_data_t, format_ldnc}, engine, hx.data_ptr());

  auto src_iter_c_usr_m = dpcpp_onednn_memory(
      {{src_iter_c_tz}, iter_c_data_t, format_ldnc}, engine, cx_.data_ptr());

  auto dst_layer_usr_m = dpcpp_onednn_memory(
      {{dst_layer_tz}, src_data_t, format_tnc}, engine, layer_y.data_ptr());

  auto dst_iter_usr_m = dpcpp_onednn_memory(
      {{dst_iter_tz}, src_data_t, format_ldnc}, engine, hy.data_ptr());

  auto dst_iter_c_usr_m = dpcpp_onednn_memory(
      {{dst_iter_c_tz}, iter_c_data_t, format_ldnc}, engine, cy.data_ptr());

  auto expected_wghs_layer_md = lstm_forward_pd->weights_layer_desc();
  auto wghs_layer_m = wghs_layer_usr_m;
  if (wghs_layer_usr_m.get_desc() != expected_wghs_layer_md) {
    wghs_layer_m = memory(expected_wghs_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(wghs_layer_usr_m, wghs_layer_m),
        strm,
        {{DNNL_ARG_FROM, wghs_layer_usr_m}, {DNNL_ARG_TO, wghs_layer_m}});
  }

  auto expected_wghs_iter_md = lstm_forward_pd->weights_iter_desc();
  auto wghs_iter_m = wghs_iter_usr_m;
  if (wghs_iter_usr_m.get_desc() != expected_wghs_iter_md) {
    wghs_iter_m = memory(expected_wghs_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(wghs_iter_usr_m, wghs_iter_m),
        strm,
        {{DNNL_ARG_FROM, wghs_iter_usr_m}, {DNNL_ARG_TO, wghs_iter_m}});
  }

  auto expected_bia_md = lstm_forward_pd->bias_desc();
  auto bia_m = bia_usr_m;
  if (bia_usr_m.get_desc() != expected_bia_md) {
    bia_m = memory(expected_bia_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bia_usr_m, bia_m),
        strm,
        {{DNNL_ARG_FROM, bia_usr_m}, {DNNL_ARG_TO, bia_m}});
  }

  auto expected_src_layer_md = lstm_forward_pd->src_layer_desc();
  auto src_layer_m = src_layer_usr_m;
  if (src_layer_usr_m.get_desc() != expected_src_layer_md) {
    src_layer_m = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_m, src_layer_m),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_m}, {DNNL_ARG_TO, src_layer_m}});
  }

  auto expected_src_iter_md = lstm_forward_pd->src_iter_desc();
  auto src_iter_m = src_iter_usr_m;
  if (src_iter_usr_m.get_desc() != expected_src_iter_md) {
    src_iter_m = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_m, src_iter_m),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_m}, {DNNL_ARG_TO, src_iter_m}});
  }

  auto expected_src_iter_c_md = lstm_forward_pd->src_iter_c_desc();
  auto src_iter_c_m = src_iter_c_usr_m;
  if (src_iter_c_usr_m.get_desc() != expected_src_iter_c_md) {
    src_iter_c_m = memory(expected_src_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_c_usr_m, src_iter_c_m),
        strm,
        {{DNNL_ARG_FROM, src_iter_c_usr_m}, {DNNL_ARG_TO, src_iter_c_m}});
  }

  auto expected_dst_layer_md = lstm_forward_pd->dst_layer_desc();
  auto dst_layer_m = dst_layer_usr_m;
  if (dst_layer_usr_m.get_desc() != expected_dst_layer_md) {
    dst_layer_m = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_m, dst_layer_m),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_m}, {DNNL_ARG_TO, dst_layer_m}});
  }

  auto expected_dst_iter_md = lstm_forward_pd->dst_iter_desc();
  auto dst_iter_m = dst_iter_usr_m;
  if (dst_iter_usr_m.get_desc() != expected_dst_iter_md) {
    dst_iter_m = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_m, dst_iter_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_m}, {DNNL_ARG_TO, dst_iter_m}});
  }

  auto expected_dst_iter_c_md = lstm_forward_pd->dst_iter_c_desc();
  auto dst_iter_c_m = dst_iter_c_usr_m;
  if (dst_iter_c_usr_m.get_desc() != expected_dst_iter_c_md) {
    dst_iter_c_m = memory(expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_usr_m, dst_iter_c_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_usr_m}, {DNNL_ARG_TO, dst_iter_c_m}});
  }

  std::shared_ptr<lstm_forward> lstm1_forward;
  lstm1_forward.reset(new lstm_forward(*lstm_forward_pd));
  if (train) {
    auto workspace_md = lstm_forward_pd->workspace_desc();
    workspace_t = at::zeros(workspace_md.get_size(), src.options());
    auto workspace =
        dpcpp_onednn_memory(workspace_md, engine, workspace_t.data_ptr());

    DPCPP_ONEDNN_EXEC(
        *lstm1_forward,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_m},
         {DNNL_ARG_SRC_ITER, src_iter_m},
         {DNNL_ARG_SRC_ITER_C, src_iter_c_m},
         {DNNL_ARG_WEIGHTS_LAYER, wghs_layer_m},
         {DNNL_ARG_WEIGHTS_ITER, wghs_iter_m},
         {DNNL_ARG_BIAS, bia_m},
         {DNNL_ARG_DST_LAYER, dst_layer_m},
         {DNNL_ARG_DST_ITER, dst_iter_m},
         {DNNL_ARG_DST_ITER_C, dst_iter_c_m},
         {DNNL_ARG_WORKSPACE, workspace}});
  } else {
    DPCPP_ONEDNN_EXEC(
        *lstm1_forward,
        strm,
        {{DNNL_ARG_SRC_LAYER, src_layer_m},
         {DNNL_ARG_SRC_ITER, src_iter_m},
         {DNNL_ARG_SRC_ITER_C, src_iter_c_m},
         {DNNL_ARG_WEIGHTS_LAYER, wghs_layer_m},
         {DNNL_ARG_WEIGHTS_ITER, wghs_iter_m},
         {DNNL_ARG_BIAS, bia_m},
         {DNNL_ARG_DST_LAYER, dst_layer_m},
         {DNNL_ARG_DST_ITER, dst_iter_m},
         {DNNL_ARG_DST_ITER_C, dst_iter_c_m}});
  }

  if (dst_layer_m != dst_layer_usr_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_m, dst_layer_usr_m),
        strm,
        {{DNNL_ARG_FROM, dst_layer_m}, {DNNL_ARG_TO, dst_layer_usr_m}});
  }

  if (dst_iter_m != dst_iter_usr_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_m, dst_iter_usr_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_m}, {DNNL_ARG_TO, dst_iter_usr_m}});
  }

  if (dst_iter_c_m != dst_iter_c_usr_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_m, dst_iter_c_usr_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_m}, {DNNL_ARG_TO, dst_iter_c_usr_m}});
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
lstm_backward(
    const Tensor& src,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& dst,
    const Tensor& hy,
    const Tensor& cy,
    const Tensor& wgh_i,
    const Tensor& wgh_h,
    const Tensor& bia,
    Tensor workspace_arr,
    const Tensor& diff_dst,
    const Tensor& diff_hy_ori,
    const Tensor& diff_cy_ori,
    int layer_num,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t hidden_sz = hx.size(-1);
  int32_t seq_length = src.size(0);
  int32_t mini_batch = src.size(1);
  int32_t src_sz = src.size(2);
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t num_gate = 4;

  Tensor diff_hy, diff_cy;
  if (diff_hy_ori.defined()) {
    diff_hy = diff_hy_ori;
  } else {
    diff_hy = at::zeros_like(hy);
  }
  if (diff_cy_ori.defined()) {
    diff_cy = diff_cy_ori;
  } else {
    diff_cy = at::zeros_like(cy);
  }

  Tensor layer_x = at::empty_like(src);
  layer_x.copy_(src);

  Tensor layer_y = at::empty_like(dst);
  layer_y.copy_(dst);

  Tensor diff_layer_y = at::empty_like(diff_dst);
  diff_layer_y.copy_(diff_dst);

  auto diff_hx = at::zeros_like(hx);
  auto diff_cx = at::zeros_like(cx);
  auto diff_wgh_i = at::zeros_like(wgh_i);
  auto diff_wgh_h = at::zeros_like(wgh_h);
  auto diff_bia = at::zeros_like(bia);

  auto src_data_t = xpu::oneDNN::get_onednn_dtype(src);
  auto iter_c_data_t = xpu::oneDNN::get_onednn_dtype(src);

  auto bia_ = bia;
  auto cx_ = cx;
  auto bia_data_t = xpu::oneDNN::get_onednn_dtype(bia_);
  auto diff_data_t = xpu::oneDNN::get_onednn_dtype(diff_dst);
  if (bia_.scalar_type() == ScalarType::BFloat16) {
    bia_ = bia_.to(at::kFloat);
    cx_ = cx_.to(at::kFloat);
    diff_layer_y = diff_layer_y.to(at::kFloat);
    bia_data_t = memory::data_type::f32;
    iter_c_data_t = memory::data_type::f32;
    diff_data_t = memory::data_type::f32;
  }

  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  memory::dims src_layer_0_tz = {
      seq_length, mini_batch, src_sz}; // for layer=0, tnc
  memory::dims src_layer_tz = {
      seq_length, mini_batch, hidden_sz * num_directions}; // for layer>0, tnc
  memory::dims src_iter_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc hx, src hidden state
  memory::dims src_iter_c_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc cx_, src cell state

  memory::dims wghs_layer_0_tz = {
      1, num_directions, src_sz, num_gate, hidden_sz}; // for layer=0, ldigo
  memory::dims wghs_layer_tz = {
      1,
      num_directions,
      hidden_sz * num_directions,
      num_gate,
      hidden_sz}; // for layer>0, ldigo
  memory::dims wghs_iter_tz = {
      1,
      num_directions,
      hidden_sz,
      num_gate,
      hidden_sz}; // ldigo wgh for hidden
  memory::dims bia_tz = {1, num_directions, num_gate, hidden_sz}; // ldgo

  memory::dims dst_layer_tz = {
      seq_length, mini_batch, hidden_sz * num_directions}; // tnc
  memory::dims dst_iter_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc   hy, dst hidden state
  memory::dims dst_iter_c_tz = {
      1, num_directions, mini_batch, hidden_sz}; // ldnc  cy, dst cell state

  auto diff_src = at::zeros_like(src);
  int i = layer_num;
  auto diff_layer_x = at::zeros(
      {i == 0 ? src_layer_0_tz : src_layer_tz}, diff_layer_y.options());

  auto src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_tz : src_layer_tz}, src_data_t, format_any);
  auto wghs_layer_md = memory::desc(
      {i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, src_data_t, format_any);
  auto wghs_iter_md = memory::desc({wghs_iter_tz}, src_data_t, format_any);
  auto bia_md = memory::desc({bia_tz}, bia_data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_tz}, src_data_t, format_any);
  auto src_iter_c_md = memory::desc({src_iter_c_tz}, iter_c_data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_tz}, src_data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_tz}, src_data_t, format_any);
  auto dst_iter_c_md = memory::desc({dst_iter_c_tz}, iter_c_data_t, format_any);
  auto diff_src_layer_md = memory::desc(
      {i == 0 ? src_layer_0_tz : src_layer_tz}, diff_data_t, format_any);
  auto diff_src_iter_md = memory::desc({src_iter_tz}, diff_data_t, format_any);
  auto diff_src_iter_c_md =
      memory::desc({src_iter_c_tz}, diff_data_t, format_any);
  auto diff_wghs_layer_md = memory::desc(
      {i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, diff_data_t, format_any);
  auto diff_wghs_iter_md =
      memory::desc({wghs_iter_tz}, diff_data_t, format_any);
  auto diff_bia_md = memory::desc({bia_tz}, diff_data_t, format_any);
  auto diff_dst_layer_md =
      memory::desc({dst_layer_tz}, diff_data_t, format_any);
  auto diff_dst_iter_md = memory::desc({dst_iter_tz}, diff_data_t, format_any);
  auto diff_dst_iter_c_md =
      memory::desc({dst_iter_c_tz}, diff_data_t, format_any);

  rnn_direction dir = bidirectional ? rnn_direction::bidirectional_concat
                                    : rnn_direction::unidirectional_left2right;

  std::shared_ptr<lstm_forward::desc> lstm_forward_desc;
  lstm_forward_desc.reset(new lstm_forward::desc(
      prop_kind::forward_training,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      wghs_layer_md,
      wghs_iter_md,
      bia_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md));

  std::shared_ptr<lstm_forward::primitive_desc> lstm_forward_pd;
  lstm_forward_pd.reset(
      new lstm_forward::primitive_desc(*lstm_forward_desc, engine));

  std::shared_ptr<dnnl::lstm_backward::desc> lstm_backward_desc;
  lstm_backward_desc.reset(new dnnl::lstm_backward::desc(
      prop_kind::backward,
      dir,
      src_layer_md,
      src_iter_md,
      src_iter_c_md,
      wghs_layer_md,
      wghs_iter_md,
      bia_md,
      dst_layer_md,
      dst_iter_md,
      dst_iter_c_md,
      diff_src_layer_md,
      diff_src_iter_md,
      diff_src_iter_c_md,
      diff_wghs_layer_md,
      diff_wghs_iter_md,
      diff_bia_md,
      diff_dst_layer_md,
      diff_dst_iter_md,
      diff_dst_iter_c_md));

  auto src_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_tz : src_layer_tz}, src_data_t, format_tnc},
      engine,
      layer_x.data_ptr());

  auto src_iter_usr_m = dpcpp_onednn_memory(
      {{src_iter_tz}, src_data_t, format_ldnc}, engine, hx.data_ptr());

  auto src_iter_c_usr_m = dpcpp_onednn_memory(
      {{src_iter_c_tz}, iter_c_data_t, format_ldnc}, engine, cx_.data_ptr());

  auto dst_layer_usr_m = dpcpp_onednn_memory(
      {{dst_layer_tz}, src_data_t, format_tnc}, engine, layer_y.data_ptr());

  auto dst_iter_usr_m = dpcpp_onednn_memory(
      {{dst_iter_tz}, src_data_t, format_ldnc}, engine, hy.data_ptr());

  auto dst_iter_c_usr_m = dpcpp_onednn_memory(
      {{dst_iter_c_tz}, iter_c_data_t, format_ldnc}, engine, cy.data_ptr());

  auto wghs_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, src_data_t, format_ldigo},
      engine,
      wgh_i.data_ptr());

  auto wghs_iter_usr_m = dpcpp_onednn_memory(
      {{wghs_iter_tz}, src_data_t, format_ldigo}, engine, wgh_h.data_ptr());

  auto bia_usr_m = dpcpp_onednn_memory(
      {{bia_tz}, bia_data_t, format_ldgo}, engine, bia_.data_ptr());

  auto diff_src_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? src_layer_0_tz : src_layer_tz}, diff_data_t, format_tnc},
      engine,
      diff_layer_x.data_ptr());

  auto diff_src_iter_usr_m = dpcpp_onednn_memory(
      {{src_iter_tz}, diff_data_t, format_ldnc}, engine, diff_hx.data_ptr());

  auto diff_src_iter_c_usr_m = dpcpp_onednn_memory(
      {{src_iter_c_tz}, diff_data_t, format_ldnc}, engine, diff_cx.data_ptr());

  auto diff_dst_layer_usr_m = dpcpp_onednn_memory(
      {{dst_layer_tz}, diff_data_t, format_tnc},
      engine,
      diff_layer_y.data_ptr());

  auto diff_dst_iter_usr_m = dpcpp_onednn_memory(
      {{dst_iter_tz}, diff_data_t, format_ldnc}, engine, diff_hy.data_ptr());

  auto diff_dst_iter_c_usr_m = dpcpp_onednn_memory(
      {{dst_iter_c_tz}, diff_data_t, format_ldnc}, engine, diff_cy.data_ptr());

  auto diff_wghs_layer_usr_m = dpcpp_onednn_memory(
      {{i == 0 ? wghs_layer_0_tz : wghs_layer_tz}, diff_data_t, format_ldigo},
      engine,
      diff_wgh_i.data_ptr());

  auto diff_wghs_iter_usr_m = dpcpp_onednn_memory(
      {{wghs_iter_tz}, diff_data_t, format_ldigo},
      engine,
      diff_wgh_h.data_ptr());

  auto diff_bia_usr_m = dpcpp_onednn_memory(
      {{bia_tz}, diff_data_t, format_ldgo}, engine, diff_bia.data_ptr());

  std::shared_ptr<dnnl::lstm_backward::primitive_desc> lstm_backward_pd;
  lstm_backward_pd.reset(new dnnl::lstm_backward::primitive_desc(
      *lstm_backward_desc, engine, *lstm_forward_pd));

  auto expected_src_layer_md = lstm_forward_pd->src_layer_desc();
  auto src_layer_m = src_layer_usr_m;
  if (src_layer_usr_m.get_desc() != expected_src_layer_md) {
    src_layer_m = memory(expected_src_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_layer_usr_m, src_layer_m),
        strm,
        {{DNNL_ARG_FROM, src_layer_usr_m}, {DNNL_ARG_TO, src_layer_m}});
  }

  auto expected_src_iter_md = lstm_forward_pd->src_iter_desc();
  auto src_iter_m = src_iter_usr_m;
  if (src_iter_usr_m.get_desc() != expected_src_iter_md) {
    src_iter_m = memory(expected_src_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_usr_m, src_iter_m),
        strm,
        {{DNNL_ARG_FROM, src_iter_usr_m}, {DNNL_ARG_TO, src_iter_m}});
  }

  auto expected_src_iter_c_md = lstm_forward_pd->src_iter_c_desc();
  auto src_iter_c_m = src_iter_c_usr_m;
  if (src_iter_c_usr_m.get_desc() != expected_src_iter_c_md) {
    src_iter_c_m = memory(expected_src_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(src_iter_c_usr_m, src_iter_c_m),
        strm,
        {{DNNL_ARG_FROM, src_iter_c_usr_m}, {DNNL_ARG_TO, src_iter_c_m}});
  }

  auto expected_dst_layer_md = lstm_forward_pd->dst_layer_desc();
  auto dst_layer_m = dst_layer_usr_m;
  if (dst_layer_usr_m.get_desc() != expected_dst_layer_md) {
    dst_layer_m = memory(expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_layer_usr_m, dst_layer_m),
        strm,
        {{DNNL_ARG_FROM, dst_layer_usr_m}, {DNNL_ARG_TO, dst_layer_m}});
  }

  auto expected_dst_iter_md = lstm_forward_pd->dst_iter_desc();
  auto dst_iter_m = dst_iter_usr_m;
  if (dst_iter_usr_m.get_desc() != expected_dst_iter_md) {
    dst_iter_m = memory(expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_usr_m, dst_iter_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_usr_m}, {DNNL_ARG_TO, dst_iter_m}});
  }

  auto expected_dst_iter_c_md = lstm_forward_pd->dst_iter_c_desc();
  auto dst_iter_c_m = dst_iter_c_usr_m;
  if (dst_iter_c_usr_m.get_desc() != expected_dst_iter_c_md) {
    dst_iter_c_m = memory(expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(dst_iter_c_usr_m, dst_iter_c_m),
        strm,
        {{DNNL_ARG_FROM, dst_iter_c_usr_m}, {DNNL_ARG_TO, dst_iter_c_m}});
  }

  auto expected_bia_md = lstm_backward_pd->bias_desc();
  auto bia_m = bia_usr_m;
  if (bia_usr_m.get_desc() != expected_bia_md) {
    bia_m = memory(expected_bia_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(bia_usr_m, bia_m),
        strm,
        {{DNNL_ARG_FROM, bia_usr_m}, {DNNL_ARG_TO, bia_m}});
  }

  auto expected_bwd_wghs_layer_md = lstm_backward_pd->weights_layer_desc();
  auto bwd_wghs_layer_m = wghs_layer_usr_m;
  if (wghs_layer_usr_m.get_desc() != expected_bwd_wghs_layer_md) {
    bwd_wghs_layer_m = memory(expected_bwd_wghs_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(wghs_layer_usr_m, bwd_wghs_layer_m),
        strm,
        {{DNNL_ARG_FROM, wghs_layer_usr_m}, {DNNL_ARG_TO, bwd_wghs_layer_m}});
  }

  auto expected_bwd_wghs_iter_md = lstm_backward_pd->weights_iter_desc();
  auto bwd_wghs_iter_m = wghs_iter_usr_m;
  if (wghs_iter_usr_m.get_desc() != expected_bwd_wghs_iter_md) {
    bwd_wghs_iter_m = memory(expected_bwd_wghs_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(wghs_iter_usr_m, bwd_wghs_iter_m),
        strm,
        {{DNNL_ARG_FROM, wghs_iter_usr_m}, {DNNL_ARG_TO, bwd_wghs_iter_m}});
  }

  auto diff_expected_src_layer_md = lstm_backward_pd->diff_src_layer_desc();
  auto diff_src_layer_m = diff_src_layer_usr_m;
  if (diff_src_layer_usr_m.get_desc() != diff_expected_src_layer_md) {
    diff_src_layer_m = memory(diff_expected_src_layer_md, engine);
  }

  auto diff_expected_src_iter_md = lstm_backward_pd->diff_src_iter_desc();
  auto diff_src_iter_m = diff_src_iter_usr_m;
  if (diff_src_iter_usr_m.get_desc() != diff_expected_src_iter_md) {
    diff_src_iter_m = memory(diff_expected_src_iter_md, engine);
  }

  auto diff_expected_src_iter_c_md = lstm_backward_pd->diff_src_iter_c_desc();
  auto diff_src_iter_c_m = diff_src_iter_c_usr_m;
  if (diff_src_iter_c_usr_m.get_desc() != diff_expected_src_iter_c_md) {
    diff_src_iter_c_m = memory(diff_expected_src_iter_c_md, engine);
  }

  auto diff_expected_dst_layer_md = lstm_backward_pd->diff_dst_layer_desc();
  auto diff_dst_layer_m = diff_dst_layer_usr_m;
  if (diff_dst_layer_usr_m.get_desc() != diff_expected_dst_layer_md) {
    diff_dst_layer_m = memory(diff_expected_dst_layer_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_dst_layer_usr_m, diff_dst_layer_m),
        strm,
        {{DNNL_ARG_FROM, diff_dst_layer_usr_m},
         {DNNL_ARG_TO, diff_dst_layer_m}});
  }

  auto diff_expected_dst_iter_md = lstm_backward_pd->diff_dst_iter_desc();
  auto diff_dst_iter_m = diff_dst_iter_usr_m;
  if (diff_dst_iter_usr_m.get_desc() != diff_expected_dst_iter_md) {
    diff_dst_iter_m = memory(diff_expected_dst_iter_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_dst_iter_usr_m, diff_dst_iter_m),
        strm,
        {{DNNL_ARG_FROM, diff_dst_iter_usr_m}, {DNNL_ARG_TO, diff_dst_iter_m}});
  }

  auto diff_expected_dst_iter_c_md = lstm_backward_pd->diff_dst_iter_c_desc();
  auto diff_dst_iter_c_m = diff_dst_iter_c_usr_m;
  if (diff_dst_iter_c_usr_m.get_desc() != diff_expected_dst_iter_c_md) {
    diff_dst_iter_c_m = memory(diff_expected_dst_iter_c_md, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_dst_iter_c_usr_m, diff_dst_iter_c_m),
        strm,
        {{DNNL_ARG_FROM, diff_dst_iter_c_usr_m},
         {DNNL_ARG_TO, diff_dst_iter_c_m}});
  }

  auto diff_expected_wghs_layer_md =
      lstm_backward_pd->diff_weights_layer_desc();
  auto diff_wghs_layer_m = diff_wghs_layer_usr_m;
  if (diff_wghs_layer_usr_m.get_desc() != diff_expected_wghs_layer_md) {
    diff_wghs_layer_m = memory(diff_expected_wghs_layer_md, engine);
  }

  auto diff_expected_wghs_iter_md = lstm_backward_pd->diff_weights_iter_desc();
  auto diff_wghs_iter_m = diff_wghs_iter_usr_m;
  if (diff_wghs_iter_usr_m.get_desc() != diff_expected_wghs_iter_md) {
    diff_wghs_iter_m = memory(diff_expected_wghs_iter_md, engine);
  }

  auto diff_expected_bia_md = lstm_backward_pd->diff_bias_desc();
  auto diff_bia_m = diff_bia_usr_m;
  if (diff_bia_usr_m.get_desc() != diff_expected_bia_md) {
    diff_bia_m = memory(diff_expected_bia_md, engine);
  }

  auto workspace = dpcpp_onednn_memory(
      lstm_backward_pd->workspace_desc(), engine, workspace_arr.data_ptr());

  std::shared_ptr<dnnl::lstm_backward> lstm_backward_p;
  lstm_backward_p.reset(new dnnl::lstm_backward(*lstm_backward_pd));

  DPCPP_ONEDNN_EXEC(
      *lstm_backward_p,
      strm,
      {{DNNL_ARG_SRC_LAYER, src_layer_m},
       {DNNL_ARG_SRC_ITER, src_iter_m},
       {DNNL_ARG_SRC_ITER_C, src_iter_c_m},
       {DNNL_ARG_WEIGHTS_LAYER, bwd_wghs_layer_m},
       {DNNL_ARG_WEIGHTS_ITER, bwd_wghs_iter_m},
       {DNNL_ARG_BIAS, bia_m},
       {DNNL_ARG_DST_LAYER, dst_layer_m},
       {DNNL_ARG_DST_ITER, dst_iter_m},
       {DNNL_ARG_DST_ITER_C, dst_iter_c_m},
       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_m},
       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_m},
       {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c_m},
       {DNNL_ARG_WORKSPACE, workspace},
       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_m},
       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_m},
       {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c_m},
       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_wghs_layer_m},
       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_wghs_iter_m},
       {DNNL_ARG_DIFF_BIAS, diff_bia_m}});

  if (diff_src_layer_usr_m != diff_src_layer_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_src_layer_m, diff_src_layer_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_src_layer_m},
         {DNNL_ARG_TO, diff_src_layer_usr_m}});
  }
  if (diff_src_iter_usr_m != diff_src_iter_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_src_iter_m, diff_src_iter_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_src_iter_m}, {DNNL_ARG_TO, diff_src_iter_usr_m}});
  }
  if (diff_src_iter_c_usr_m != diff_src_iter_c_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_src_iter_c_m, diff_src_iter_c_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_src_iter_c_m},
         {DNNL_ARG_TO, diff_src_iter_c_usr_m}});
  }
  if (diff_wghs_layer_usr_m != diff_wghs_layer_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_wghs_layer_m, diff_wghs_layer_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_wghs_layer_m},
         {DNNL_ARG_TO, diff_wghs_layer_usr_m}});
  }
  if (diff_wghs_iter_usr_m != diff_wghs_iter_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_wghs_iter_m, diff_wghs_iter_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_wghs_iter_m},
         {DNNL_ARG_TO, diff_wghs_iter_usr_m}});
  }
  if (diff_bia_usr_m != diff_bia_m) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(diff_bia_m, diff_bia_usr_m),
        strm,
        {{DNNL_ARG_FROM, diff_bia_m}, {DNNL_ARG_TO, diff_bia_usr_m}});
  }
  diff_src = diff_layer_x;

  return std::tuple<
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor>{diff_src, diff_hx, diff_cx, diff_wgh_i, diff_wgh_h, diff_bia};
}

} // namespace oneDNN
} // namespace xpu
