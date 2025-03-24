/*******************************************************************************
 * Copyright (C) 2025 Intel Corporation
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission. This software and the related documents are provided as is,
 * with no express or implied warranties, other than those that are expressly
 * stated in the License.
 *******************************************************************************
 */
#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include "Attr.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace torch_ipex::xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace torch_ipex::xpu {
namespace oneDNN {
static inline void fp8_matmul(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& b_raw,
    const Tensor& m1_sc,
    const Tensor& m2_sc) {
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  size_t dims = result.dim();
  Tensor m1 = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(mat1)
      ? mat1
      : contiguous_if_needed(mat1);
  Tensor m2 = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(mat2)
      ? mat2
      : contiguous_if_needed(mat2);
  Tensor dst = result;
  Tensor b = b_raw;

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;
  if (dims == 3) {
    mb = dst.size(0);
  }

  // validate bias and make it compatible with oneDNN implementation
  bool with_bias = b.defined() ? true : false;
  if (with_bias) {
    if (b.dim() == 1) {
      TORCH_CHECK(
          b.size(0) == n || b.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1 ...");
      if (b.size(0) == 0) {
        with_bias = false;
      } else if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else if (m1.dim() == 2) {
        b = b.expand({1, n}).contiguous();
      }
    } else if (b.dim() == 2) {
      TORCH_CHECK(
          (b.size(0) == m && b.size(1) == n) ||
              (b.size(0) == 1 && b.size(1) == n) ||
              (b.size(0) == m && b.size(1) == 1) ||
              (b.size(0) == 1 && b.size(1) == 1),
          "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
      if (b.size(0) == 1 && b.size(1) == 1)
        b = b.expand({1, n}).contiguous();
    } else if (b.dim() == 3) {
      TORCH_CHECK(
          are_expandable({mb, m, n}, b.sizes()),
          "matmul bias must be expandable to:",
          dst.sizes(),
          " but got:",
          b.sizes());
      b = b.expand({mb, m, n}).contiguous();
    } else if (b.dim() == 0) {
      TORCH_CHECK(
          b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else {
        b = b.expand({1, n}).contiguous();
      }
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
    }
  }

  auto m1_usr_dt = get_onednn_dtype(m1);
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto dst_usr_dt = get_onednn_dtype(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;
  memory::data_type bias_dt;

  memory::desc m1_md, m1_usr_md;
  memory::desc m2_md, m2_usr_md;
  memory::desc dst_md, dst_usr_md;
  memory::desc b_md;

  memory::dims m1_dims, m2_dims, dst_dims, bias_dims;
  memory::dims m1_strides, m2_strides, dst_strides, bias_strides;
  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    m2_strides = {m2.stride(0), m2.stride(1)};
    dst_strides = {dst.stride(0), dst.stride(1)};
  } else {
    m1_dims = {mb, m, k};
    m2_dims = {mb, k, n};
    dst_dims = {mb, m, n};

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    m2_strides = {m2.stride(0), m2.stride(1), m2.stride(2)};
    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  if (with_bias) {
    bias_dims = get_onednn_dims(b);
    bias_dt = get_onednn_dtype(b);
    bias_strides = get_onednn_strides(b);
  }

  std::unordered_map<int, memory> args;
  m1_md = memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = memory::desc(m2_dims, m2_dt, m2_strides);
  dst_md = memory::desc(dst_dims, dst_dt, dst_strides);

  primitive_attr pattr;
  memory m1_sc_m =
      dpcpp_onednn_memory(Q_PER_TENSOR_SC_MD, engine, m1_sc.data_ptr<float>());
  memory m2_sc_m =
      dpcpp_onednn_memory(Q_PER_TENSOR_SC_MD, engine, m2_sc.data_ptr<float>());

  pattr.set_scales_mask(DNNL_ARG_SRC, 0);
  pattr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);

  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;
  if (with_bias) {
    b_md = memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd =
        matmul::primitive_desc(engine, m1_md, m2_md, b_md, dst_md, pattr);
  } else {
    matmul_pd = matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }

  matmul_p = dnnl::matmul(matmul_pd);

  m1_usr_md = memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = memory::desc(m2_dims, m2_usr_dt, m2_strides);
  dst_usr_md = memory::desc(dst_dims, dst_usr_dt, dst_strides);

  auto m1_usr_m = dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  auto dst_usr_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  Tensor m1_, m2_, dst_;

  if (m1_usr_m.get_desc() != expected_m1_md) {
    m1_ = empty_opaque_tensor(expected_m1_md, m1.options(), c10::nullopt);
    m1_m = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
    torch_ipex::xpu::oneDNN::reorder(m1, m1_);
  }

  if (m2_usr_m.get_desc() != expected_m2_md) {
    m2_ = empty_opaque_tensor(expected_m2_md, m2.options(), c10::nullopt);
    m2_m = dpcpp_onednn_memory(expected_m2_md, engine, m2_.data_ptr());
    torch_ipex::xpu::oneDNN::reorder(m2, m2_);
  }

  if (dst_usr_m.get_desc() != expected_dst_md) {
    dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());
  }

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc_m});
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m2_sc_m});
  if (with_bias) {
    auto b_m = dpcpp_onednn_memory(b_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, b_m});
  }

  DPCPP_ONEDNN_EXEC(matmul_p, strm, args);

  if (!dst.is_same(result))
    result.copy_(dst);
}

} // namespace oneDNN
} // namespace torch_ipex::xpu
