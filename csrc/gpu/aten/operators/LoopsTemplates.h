#pragma once

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

template <dnnl::algorithm algorithm_t, typename iter_creator_t, typename func_t>
static inline Tensor& unary_out_with_onednn_and_loops(
    iter_creator_t iter_creator,
    Tensor& out,
    const Tensor& self,
    func_t fn,
    float alpha = 0.0f,
    float beta = 0.0f,
    bool if_enable_onednn_path = true) {
  bool is_out_defined = out.defined();
  bool use_onednn_path =
      if_enable_onednn_path && xpu::oneDNN::eltwise_forward_valid(out, self);

  Tensor self_;
  if (use_onednn_path) {
    self_ = self;
  } else {
    self_ = at::AtenIpexTypeXPU::to_plain_if_needed(self);
    if (is_out_defined)
      out = at::AtenIpexTypeXPU::to_plain_if_needed_(out);
  }

  auto iter = iter_creator(out, self_);

  if (use_onednn_path) {
    xpu::oneDNN::eltwise<algorithm_t>(out, self_, alpha, beta);
  } else {
    fn(iter);
    if (!is_out_defined)
      out = iter.output();
  }

  return out;
}

template <dnnl::algorithm algorithm_t, typename iter_creator_t, typename func_t>
static inline Tensor& unary_out_with_onednn_and_loops_bw(
    iter_creator_t iter_creator,
    Tensor& out,
    const Tensor& self,
    const Tensor& other,
    func_t fn,
    float alpha = 0.0f,
    float beta = 0.0f,
    bool if_enable_onednn_path = true) {
  bool is_out_defined = out.defined();
  bool use_onednn_path = if_enable_onednn_path &&
      xpu::oneDNN::eltwise_backward_valid(out, self, other);

  Tensor self_, other_;
  if (use_onednn_path) {
    self_ = self;
    other_ = other;
  } else {
    self_ = at::AtenIpexTypeXPU::to_plain_if_needed(self);
    other_ = at::AtenIpexTypeXPU::to_plain_if_needed(other);
    if (is_out_defined)
      out = at::AtenIpexTypeXPU::to_plain_if_needed_(out);
  }

  auto iter = iter_creator(out, self_, other_);

  if (use_onednn_path) {
    xpu::oneDNN::eltwise_backward<algorithm_t>(out, self_, other_, alpha, beta);
  } else {
    fn(iter);
    if (!is_out_defined)
      out = iter.output();
  }

  return out;
}

template <dnnl::algorithm algorithm_t, typename iter_creator_t, typename func_t>
static inline Tensor& binary_out_template(
    iter_creator_t iter_creator,
    Tensor& out,
    const Tensor& self,
    const Tensor& other,
    func_t fn,
    bool if_enable_onednn_path = true) {
  bool is_out_defined = out.defined();
  bool use_onednn_path = if_enable_onednn_path &&
      xpu::oneDNN::binary_forward_valid(out, self, other);

  Tensor self_, other_;
  if (use_onednn_path) {
    self_ = self;
    other_ = other;
  } else {
    self_ = at::AtenIpexTypeXPU::to_plain_if_needed(self);
    other_ = at::AtenIpexTypeXPU::to_plain_if_needed(other);
    if (is_out_defined)
      out = at::AtenIpexTypeXPU::to_plain_if_needed_(out);
  }

  auto iter = iter_creator(out, self_, other_);

  if (use_onednn_path) {
    xpu::oneDNN::bin<algorithm_t>(out, self_, other_);
  } else {
    fn(iter);
    if (!is_out_defined)
      out = iter.output();
  }

  return out;
}
