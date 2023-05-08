#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <operators/comm/Scalar.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline Tensor gru_forward(
    const Tensor& src_layer,
    const Tensor& src_iter,
    const Tensor& weight_layer,
    const Tensor& weight_iter,
    const Tensor& bias,
    Tensor& dst_layer,
    Tensor& dst_iter,
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

  auto src_layer_md = memory::desc({src_layer_dims}, data_t, format_any);
  auto weights_layer_md =
      memory::desc({weights_layer_dims}, data_t, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_t, format_any);
  auto bias_md = memory::desc({bias_dims}, bias_data_t, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_t, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_t, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_t, format_any);

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (data_t == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

  auto gru_forward_pd = lbr_gru_forward::primitive_desc(
      engine,
      train ? prop_kind::forward_training : prop_kind::forward_inference,
      dir,
      src_layer_md,
      src_iter_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      pattr);

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

  Tensor new_weight_layer = weight_layer.reshape({weights_layer_dims}),
         new_weight_iter = weight_iter.reshape({weights_iter_dims}),
         new_bias = bias.reshape({bias_dims}),
         new_src_iter = src_iter.reshape({src_iter_dims});

  Tensor weight_layer_, weight_iter_, bias_, src_layer_, src_iter_, dst_layer_,
      dst_iter_;

  auto expected_weights_layer_md = gru_forward_pd.weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weight_layer_ = empty_opaque_tensor(
        expected_weights_layer_md, weight_layer.options(), c10::nullopt);
    weights_layer_memory = dpcpp_onednn_memory(
        expected_weights_layer_md, engine, weight_layer_.data_ptr());
    xpu::oneDNN::reorder(new_weight_layer, weight_layer_);
  }

  auto expected_weights_iter_md = gru_forward_pd.weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weight_iter_ = empty_opaque_tensor(
        expected_weights_iter_md, weight_iter.options(), c10::nullopt);
    weights_iter_memory = dpcpp_onednn_memory(
        expected_weights_iter_md, engine, weight_iter_.data_ptr());
    xpu::oneDNN::reorder(new_weight_iter, weight_iter_);
  }

  auto expected_bias_md = gru_forward_pd.bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_ = empty_opaque_tensor(expected_bias_md, bias.options(), c10::nullopt);
    bias_memory =
        dpcpp_onednn_memory(expected_bias_md, engine, bias_.data_ptr());
    xpu::oneDNN::reorder(new_bias, bias_);
  }

  auto expected_src_layer_md = gru_forward_pd.src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_ = empty_opaque_tensor(
        expected_src_layer_md, src_layer.options(), c10::nullopt);
    src_layer_memory = dpcpp_onednn_memory(
        expected_src_layer_md, engine, src_layer_.data_ptr());
    xpu::oneDNN::reorder(src_layer, src_layer_);
  }

  auto expected_src_iter_md = gru_forward_pd.src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_ = empty_opaque_tensor(
        expected_src_iter_md, src_iter.options(), c10::nullopt);
    src_iter_memory =
        dpcpp_onednn_memory(expected_src_iter_md, engine, src_iter_.data_ptr());
    xpu::oneDNN::reorder(new_src_iter, src_iter_);
  }

  auto expected_dst_layer_md = gru_forward_pd.dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_ = empty_opaque_tensor(
        expected_dst_layer_md, dst_layer.options(), c10::nullopt);
    dst_layer_memory = dpcpp_onednn_memory(
        expected_dst_layer_md, engine, dst_layer_.data_ptr());
    xpu::oneDNN::reorder(dst_layer, dst_layer_);
  }

  auto expected_dst_iter_md = gru_forward_pd.dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_ = empty_opaque_tensor(
        expected_dst_iter_md, dst_iter.options(), c10::nullopt);
    dst_iter_memory =
        dpcpp_onednn_memory(expected_dst_iter_md, engine, dst_iter_.data_ptr());
    xpu::oneDNN::reorder(dst_iter, dst_iter_);
  }

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC_LAYER, src_layer_memory});
  args.insert({DNNL_ARG_SRC_ITER, src_iter_memory});
  args.insert({DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory});
  args.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_memory});
  args.insert({DNNL_ARG_BIAS, bias_memory});
  args.insert({DNNL_ARG_DST_LAYER, dst_layer_memory});
  args.insert({DNNL_ARG_DST_ITER, dst_iter_memory});

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = gru_forward_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src_layer.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      gru_forward_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  Tensor workspace;
  if (train) {
    auto workspace_md = gru_forward_pd.workspace_desc();
    auto w_size = workspace_md.get_size();
    workspace = at::empty(w_size, src_layer.options().dtype(at::kByte));
    auto workspace_memory =
        dpcpp_onednn_memory(workspace_md, engine, workspace.data_ptr());
    args.insert({DNNL_ARG_WORKSPACE, workspace_memory});
  }

  auto gru_forward_p = lbr_gru_forward(gru_forward_pd);
  DPCPP_ONEDNN_EXEC(gru_forward_p, strm, args);

  if (dst_layer_memory != dst_layer_usr_memory) {
    xpu::oneDNN::reorder(dst_layer_, dst_layer);
  }

  if (dst_iter_memory != dst_iter_usr_memory) {
    xpu::oneDNN::reorder(dst_iter_, dst_iter);
  }
  return workspace;
}

static inline std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> gru_backward(
    const Tensor& src_layer,
    const Tensor& src_iter,
    const Tensor& weight_layer,
    const Tensor& weight_iter,
    const Tensor& bias, // fp32
    const Tensor& dst_layer,
    const Tensor& dst_iter,
    const Tensor& workspace,
    const Tensor& diff_dst_layer_ori, // same dtype as src_layer
    const Tensor& diff_dst_iter_ori, // same dtype as src_layer
    bool reverse,
    int64_t hidden_size,
    bool has_bias,
    bool train,
    bool bidirectional) {
  // In backward propagation, onednn RNN needs all diff_* tensors are in f32
  // refer to https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html
  Tensor diff_dst_layer = src_layer.dtype() == at::ScalarType::BFloat16
      ? diff_dst_layer_ori.to(at::ScalarType::Float)
      : diff_dst_layer_ori;
  Tensor diff_dst_iter = src_layer.dtype() == at::ScalarType::BFloat16
      ? diff_dst_iter_ori.to(at::ScalarType::Float)
      : diff_dst_iter_ori;
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t seq_length = src_layer.size(0);
  int32_t mini_batch = src_layer.size(1);
  int32_t input_size = src_layer.size(2);
  int32_t num_gate = 3;
  int32_t num_bias_gate = 4;

  auto data_dt = get_onednn_dtype(src_layer);
  auto data_f32_dt = memory::data_type::f32;
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

  // all diff_* tensors need to be f32
  auto diff_src_layer = at::empty(
      src_layer_dims, src_layer.options().dtype(at::ScalarType::Float));
  auto diff_src_iter =
      at::empty(src_iter_dims, src_iter.options().dtype(at::ScalarType::Float));
  // onednn rnn primitive used accumulate grad, so we must zero grad buffer.
  auto diff_weight_layer = at::zeros(
      weights_layer_dims, weight_layer.options().dtype(at::ScalarType::Float));
  auto diff_weight_iter = at::zeros(
      weights_iter_dims, weight_iter.options().dtype(at::ScalarType::Float));
  auto diff_bias = at::zeros(bias_dims, bias.options());

  auto src_layer_md = memory::desc({src_layer_dims}, data_dt, format_any);
  auto weights_layer_md =
      memory::desc({weights_layer_dims}, data_dt, format_any);
  auto weights_iter_md = memory::desc({weights_iter_dims}, data_dt, format_any);
  auto bias_md = memory::desc({bias_dims}, data_f32_dt, format_any);
  auto src_iter_md = memory::desc({src_iter_dims}, data_dt, format_any);
  auto dst_layer_md = memory::desc({dst_layer_dims}, data_dt, format_any);
  auto dst_iter_md = memory::desc({dst_iter_dims}, data_dt, format_any);
  auto diff_src_layer_md =
      memory::desc({src_layer_dims}, data_f32_dt, format_any);
  auto diff_src_iter_md =
      memory::desc({src_iter_dims}, data_f32_dt, format_any);
  auto diff_weights_layer_md =
      memory::desc({weights_layer_dims}, data_f32_dt, format_any);
  auto diff_weights_iter_md =
      memory::desc({weights_iter_dims}, data_f32_dt, format_any);
  auto diff_bias_md = memory::desc({bias_dims}, data_f32_dt, format_any);
  auto diff_dst_layer_md =
      memory::desc({dst_layer_dims}, data_f32_dt, format_any);
  auto diff_dst_iter_md =
      memory::desc({dst_iter_dims}, data_f32_dt, format_any);
  rnn_direction dir = reverse ? rnn_direction::unidirectional_right2left
                              : rnn_direction::unidirectional_left2right;

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (data_dt == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

  auto gru_forward_pd = lbr_gru_forward::primitive_desc(
      engine,
      prop_kind::forward_training,
      dir,
      src_layer_md,
      src_iter_md,
      weights_layer_md,
      weights_iter_md,
      bias_md,
      dst_layer_md,
      dst_iter_md,
      pattr);

  auto src_layer_usr_memory = dpcpp_onednn_memory(
      {{src_layer_dims}, data_dt, format_tnc}, engine, src_layer.data_ptr());

  auto src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_dt, format_ldnc}, engine, src_iter.data_ptr());

  auto dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_dt, format_tnc}, engine, dst_layer.data_ptr());

  auto dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_dt, format_ldnc}, engine, dst_iter.data_ptr());

  auto weights_layer_usr_memory = dpcpp_onednn_memory(
      {{weights_layer_dims}, data_dt, format_ldigo},
      engine,
      weight_layer.data_ptr());

  auto weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_dt, format_ldigo},
      engine,
      weight_iter.data_ptr());

  auto bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_f32_dt, format_ldgo}, engine, bias.data_ptr());

  auto diff_src_layer_usr_memory = dpcpp_onednn_memory(
      {{src_layer_dims}, data_f32_dt, format_tnc},
      engine,
      diff_src_layer.data_ptr());

  auto diff_src_iter_usr_memory = dpcpp_onednn_memory(
      {{src_iter_dims}, data_f32_dt, format_ldnc},
      engine,
      diff_src_iter.data_ptr());

  auto diff_dst_layer_usr_memory = dpcpp_onednn_memory(
      {{dst_layer_dims}, data_f32_dt, format_tnc},
      engine,
      diff_dst_layer.data_ptr());

  auto diff_dst_iter_usr_memory = dpcpp_onednn_memory(
      {{dst_iter_dims}, data_f32_dt, format_ldnc},
      engine,
      diff_dst_iter.data_ptr());

  auto diff_weights_layer_usr_memory = dpcpp_onednn_memory(
      {{weights_layer_dims}, data_f32_dt, format_ldigo},
      engine,
      diff_weight_layer.data_ptr());

  auto diff_weights_iter_usr_memory = dpcpp_onednn_memory(
      {{weights_iter_dims}, data_f32_dt, format_ldigo},
      engine,
      diff_weight_iter.data_ptr());

  auto diff_bias_usr_memory = dpcpp_onednn_memory(
      {{bias_dims}, data_f32_dt, format_ldgo}, engine, diff_bias.data_ptr());

  auto gru_backward_pd = lbr_gru_backward::primitive_desc(
      engine,
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
      diff_dst_iter_md,
      gru_forward_pd,
      pattr);

  Tensor new_src_iter = src_iter.reshape({src_iter_dims}),
         new_dst_iter = dst_iter.reshape({dst_iter_dims}),
         new_weight_layer = weight_layer.reshape({weights_layer_dims}),
         new_weight_iter = weight_iter.reshape({weights_iter_dims}),
         new_bias = bias.reshape({bias_dims}),
         new_diff_dst_iter = diff_dst_iter.reshape({dst_iter_dims});

  Tensor src_layer_, src_iter_, dst_layer_, dst_iter_, weight_layer_,
      weight_iter_, bias_, bwd_weight_layer_, bwd_weight_iter_, diff_src_layer_,
      diff_src_iter_, diff_dst_layer_, diff_dst_iter_, diff_weight_layer_,
      diff_weight_iter_, diff_bias_;

  auto expected_src_layer_md = gru_forward_pd.src_layer_desc();
  auto src_layer_memory = src_layer_usr_memory;
  if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
    src_layer_ = empty_opaque_tensor(
        expected_src_layer_md, src_layer.options(), c10::nullopt);
    src_layer_memory = dpcpp_onednn_memory(
        expected_src_layer_md, engine, src_layer_.data_ptr());
    xpu::oneDNN::reorder(src_layer, src_layer_);
  }

  auto expected_src_iter_md = gru_forward_pd.src_iter_desc();
  auto src_iter_memory = src_iter_usr_memory;
  if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
    src_iter_ = empty_opaque_tensor(
        expected_src_iter_md, src_iter.options(), c10::nullopt);
    src_iter_memory =
        dpcpp_onednn_memory(expected_src_iter_md, engine, src_iter_.data_ptr());
    xpu::oneDNN::reorder(new_src_iter, src_iter_);
  }

  auto expected_dst_layer_md = gru_forward_pd.dst_layer_desc();
  auto dst_layer_memory = dst_layer_usr_memory;
  if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
    dst_layer_ = empty_opaque_tensor(
        expected_dst_layer_md, dst_layer.options(), c10::nullopt);
    dst_layer_memory = dpcpp_onednn_memory(
        expected_dst_layer_md, engine, dst_layer_.data_ptr());
    xpu::oneDNN::reorder(dst_layer, dst_layer_);
  }

  auto expected_dst_iter_md = gru_forward_pd.dst_iter_desc();
  auto dst_iter_memory = dst_iter_usr_memory;
  if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
    dst_iter_ = empty_opaque_tensor(
        expected_dst_iter_md, dst_iter.options(), c10::nullopt);
    dst_iter_memory =
        dpcpp_onednn_memory(expected_dst_iter_md, engine, dst_iter_.data_ptr());
    xpu::oneDNN::reorder(new_dst_iter, dst_iter_);
  }

  auto expected_weights_layer_md = gru_forward_pd.weights_layer_desc();
  auto weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
    weight_layer_ = empty_opaque_tensor(
        expected_weights_layer_md, weight_layer.options(), c10::nullopt);
    weights_layer_memory = dpcpp_onednn_memory(
        expected_weights_layer_md, engine, weight_layer_.data_ptr());
    xpu::oneDNN::reorder(new_weight_layer, weight_layer_);
  }

  auto expected_weights_iter_md = gru_forward_pd.weights_iter_desc();
  auto weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
    weight_iter_ = empty_opaque_tensor(
        expected_weights_iter_md, weight_iter.options(), c10::nullopt);
    weights_iter_memory = dpcpp_onednn_memory(
        expected_weights_iter_md, engine, weight_iter_.data_ptr());
    xpu::oneDNN::reorder(new_weight_iter, weight_iter_);
  }

  auto expected_bias_md = gru_backward_pd.bias_desc();
  auto bias_memory = bias_usr_memory;
  if (bias_usr_memory.get_desc() != expected_bias_md) {
    bias_ = empty_opaque_tensor(expected_bias_md, bias.options(), c10::nullopt);
    bias_memory =
        dpcpp_onednn_memory(expected_bias_md, engine, bias_.data_ptr());
    xpu::oneDNN::reorder(new_bias, bias_);
  }

  auto bwd_expected_weights_layer_md = gru_backward_pd.weights_layer_desc();
  auto bwd_weights_layer_memory = weights_layer_usr_memory;
  if (weights_layer_usr_memory.get_desc() != bwd_expected_weights_layer_md) {
    bwd_weight_layer_ = empty_opaque_tensor(
        bwd_expected_weights_layer_md, weight_layer.options(), c10::nullopt);
    bwd_weights_layer_memory = dpcpp_onednn_memory(
        bwd_expected_weights_layer_md, engine, bwd_weight_layer_.data_ptr());
    xpu::oneDNN::reorder(new_weight_layer, bwd_weight_layer_);
  }

  auto bwd_expected_weights_iter_md = gru_backward_pd.weights_iter_desc();
  auto bwd_weights_iter_memory = weights_iter_usr_memory;
  if (weights_iter_usr_memory.get_desc() != bwd_expected_weights_iter_md) {
    bwd_weight_iter_ = empty_opaque_tensor(
        bwd_expected_weights_iter_md, weight_iter.options(), c10::nullopt);
    bwd_weights_iter_memory = dpcpp_onednn_memory(
        bwd_expected_weights_iter_md, engine, bwd_weight_iter_.data_ptr());
    xpu::oneDNN::reorder(new_weight_iter, bwd_weight_iter_);
  }

  auto expected_diff_src_layer_md = gru_backward_pd.diff_src_layer_desc();
  auto diff_src_layer_memory = diff_src_layer_usr_memory;
  if (diff_src_layer_usr_memory.get_desc() != expected_diff_src_layer_md) {
    diff_src_layer_ = empty_opaque_tensor(
        expected_diff_src_layer_md, diff_src_layer.options(), c10::nullopt);
    diff_src_layer_memory =
        memory(expected_diff_src_layer_md, engine, diff_src_layer_.data_ptr());
  }

  auto expected_diff_src_iter_md = gru_backward_pd.diff_src_iter_desc();
  auto diff_src_iter_memory = diff_src_iter_usr_memory;
  if (diff_src_iter_usr_memory.get_desc() != expected_diff_src_iter_md) {
    diff_src_iter_ = empty_opaque_tensor(
        expected_diff_src_iter_md, diff_src_iter.options(), c10::nullopt);
    diff_src_iter_memory =
        memory(expected_diff_src_iter_md, engine, diff_src_iter_.data_ptr());
  }

  auto expected_diff_dst_layer_md = gru_backward_pd.diff_dst_layer_desc();
  auto diff_dst_layer_memory = diff_dst_layer_usr_memory;
  if (diff_dst_layer_usr_memory.get_desc() != expected_diff_dst_layer_md) {
    diff_dst_layer_ = empty_opaque_tensor(
        expected_diff_dst_layer_md, diff_dst_layer.options(), c10::nullopt);
    diff_dst_layer_memory = dpcpp_onednn_memory(
        expected_diff_dst_layer_md, engine, diff_dst_layer_.data_ptr());
    xpu::oneDNN::reorder(diff_dst_layer, diff_dst_layer_);
  }

  auto expected_diff_dst_iter_md = gru_backward_pd.diff_dst_iter_desc();
  auto diff_dst_iter_memory = diff_dst_iter_usr_memory;
  if (diff_dst_iter_usr_memory.get_desc() != expected_diff_dst_iter_md) {
    diff_dst_iter_ = empty_opaque_tensor(
        expected_diff_dst_iter_md, diff_dst_iter.options(), c10::nullopt);
    diff_dst_iter_memory = dpcpp_onednn_memory(
        expected_diff_dst_layer_md, engine, diff_dst_iter_.data_ptr());
    xpu::oneDNN::reorder(new_diff_dst_iter, diff_dst_iter_);
  }

  auto expected_diff_weights_layer_md =
      gru_backward_pd.diff_weights_layer_desc();
  auto diff_weights_layer_memory = diff_weights_layer_usr_memory;
  if (diff_weights_layer_usr_memory.get_desc() !=
      expected_diff_weights_layer_md) {
    diff_weight_layer_ = empty_opaque_tensor(
        expected_diff_weights_layer_md,
        diff_weight_layer.options(),
        c10::nullopt);
    diff_weights_layer_memory = dpcpp_onednn_memory(
        expected_diff_weights_layer_md, engine, diff_weight_layer_.data_ptr());
    xpu::oneDNN::reorder(diff_weight_layer, diff_weight_layer_);
  }

  auto expected_diff_weights_iter_md = gru_backward_pd.diff_weights_iter_desc();
  auto diff_weights_iter_memory = diff_weights_iter_usr_memory;
  if (diff_weights_iter_usr_memory.get_desc() !=
      expected_diff_weights_iter_md) {
    diff_weight_iter_ = empty_opaque_tensor(
        expected_diff_weights_iter_md,
        diff_weight_iter.options(),
        c10::nullopt);
    diff_weights_iter_memory = dpcpp_onednn_memory(
        expected_diff_weights_iter_md, engine, diff_weight_iter_.data_ptr());
    xpu::oneDNN::reorder(diff_weight_iter, diff_weight_iter_);
  }

  auto expected_diff_bias_md = gru_backward_pd.diff_bias_desc();
  auto diff_bias_memory = diff_bias_usr_memory;
  if (diff_bias_usr_memory.get_desc() != expected_diff_bias_md) {
    diff_bias_ = empty_opaque_tensor(
        expected_diff_bias_md, diff_bias.options(), c10::nullopt);
    diff_bias_memory = dpcpp_onednn_memory(
        expected_diff_bias_md, engine, diff_bias_.data_ptr());
    xpu::oneDNN::reorder(diff_bias, diff_bias_);
  }

  auto workspace_memory = dpcpp_onednn_memory(
      gru_backward_pd.workspace_desc(), engine, workspace.data_ptr());

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC_LAYER, src_layer_memory});
  args.insert({DNNL_ARG_SRC_ITER, src_iter_memory});
  args.insert({DNNL_ARG_WEIGHTS_LAYER, bwd_weights_layer_memory});
  args.insert({DNNL_ARG_WEIGHTS_ITER, bwd_weights_iter_memory});
  args.insert({DNNL_ARG_BIAS, bias_memory});
  args.insert({DNNL_ARG_DST_LAYER, dst_layer_memory});
  args.insert({DNNL_ARG_DST_ITER, dst_iter_memory});
  args.insert({DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_memory});
  args.insert({DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_memory});
  args.insert({DNNL_ARG_WORKSPACE, workspace_memory});
  args.insert({DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_memory});
  args.insert({DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_memory});
  args.insert({DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_memory});
  args.insert({DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_memory});
  args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_memory});

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = gru_backward_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src_layer.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      gru_backward_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  auto gru_backward_p = lbr_gru_backward(gru_backward_pd);
  DPCPP_ONEDNN_EXEC(gru_backward_p, strm, args);

  if (diff_src_layer_usr_memory != diff_src_layer_memory) {
    xpu::oneDNN::reorder(diff_src_layer_, diff_src_layer);
  }
  if (diff_src_iter_usr_memory != diff_src_iter_memory) {
    xpu::oneDNN::reorder(diff_src_iter_, diff_src_iter);
  }
  if (diff_weights_layer_usr_memory != diff_weights_layer_memory) {
    xpu::oneDNN::reorder(diff_weight_layer_, diff_weight_layer);
  }
  if (diff_weights_iter_usr_memory != diff_weights_iter_memory) {
    xpu::oneDNN::reorder(diff_weight_iter_, diff_weight_iter);
  }
  if (diff_bias_usr_memory != diff_bias_memory) {
    xpu::oneDNN::reorder(diff_bias_, diff_bias);
  }

  if (src_layer.scalar_type() == ScalarType::BFloat16) {
    diff_src_layer = diff_src_layer.to(kBFloat16);
    diff_src_iter = diff_src_iter.to(kBFloat16);
    diff_weight_layer = diff_weight_layer.to(kBFloat16);
    diff_weight_iter = diff_weight_iter.to(kBFloat16);
    diff_bias = diff_bias.to(kBFloat16);
  }

  return {
      diff_src_layer,
      diff_src_iter,
      diff_weight_layer,
      diff_weight_iter,
      diff_bias};
}
} // namespace oneDNN
} // namespace xpu
