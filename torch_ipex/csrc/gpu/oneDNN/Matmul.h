#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif


using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

struct MatmulAttr {
  static const int64_t kind_with_relu = xpu::oneDNN::with_relu;
  static const int64_t kind_with_sigmoid = xpu::oneDNN::with_sigmoid;

  MatmulAttr() : alpha_(1.f), beta_(0.f), attr_(0), m2_trans_(true) {}
  MatmulAttr(float alpha, float beta, int64_t attr, bool m2_trans) :
      alpha_(alpha), beta_(beta), attr_(attr), m2_trans_(m2_trans) {}

  bool with_relu() { return attr_ & kind_with_relu; }
  bool with_sigmoid() { return attr_ & kind_with_sigmoid; }
  int64_t attr() { return attr_; }

  float alpha_;
  float beta_;
  int64_t attr_;
  bool m2_trans_;
};

static inline void matmul(Tensor& dst, const Tensor& m1,
    const Tensor& m2, const Tensor& b, MatmulAttr attr) {
  size_t dims = dst.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "oneDNN matmul only works with 2D or 3D, got ", dims);
  TORCH_CHECK(dims == m1.dim() && dims == m2.dim(), "oneDNN input matrixes must have the same ranks");

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  if (dims == 3) {
    mb = dst.size(0);
    TORCH_CHECK(mb == m1.size(0) && mb == m2.size(0), "batch size mismatch, dst mb: ",\
        mb, "m1 mb", m1.size(0), " m2 mb: ", m2.size(0));
  }
  // ipex matmul support both ab/ba shape for m2 tensor, we don't check any more

  auto m1_usr_dt = get_onednn_dtype(m1);
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto dst_usr_dt = get_onednn_dtype(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;

  memory::desc m1_md, m1_usr_md, m1_any_md;
  memory::desc m2_md, m2_usr_md, m2_any_md;
  memory::desc dst_md, dst_usr_md, dst_any_md;
  memory::desc b_md;

  // STEP1: create memory desc

  // Naive Master weight
  if (m1_dt == memory::data_type::bf16 && m2_dt == memory::data_type::f32) {
    m2_dt = memory::data_type::bf16;
    dst_dt = memory::data_type::bf16;
  } else if (m1_dt == memory::data_type::f32 && m2_dt == memory::data_type::bf16) {
    m1_dt = memory::data_type::bf16;
    dst_dt = memory::data_type::bf16;
  }

  if (dims == 2) {
    m1_md = memory::desc({m, k}, m1_dt, {m1.stride(0), m1.stride(1)});
    m2_md = attr.m2_trans_ ?
            memory::desc({k, n}, m2_dt, {m2.stride(0), m2.stride(1)}) :
            memory::desc({k, n}, m2_dt, {m2.stride(1), m2.stride(0)});
    dst_md = memory::desc({m, n}, dst_dt, {dst.stride(0), dst.stride(1)});

    m1_usr_md = memory::desc({m, k}, m1_usr_dt, {m1.stride(0), m1.stride(1)});
    m2_usr_md = attr.m2_trans_ ?
                memory::desc({k, n}, m2_usr_dt, {m2.stride(0), m2.stride(1)}) :
                memory::desc({k, n}, m2_usr_dt, {m2.stride(1), m2.stride(0)});
    dst_usr_md = memory::desc({m, n}, dst_usr_dt, {dst.stride(0), dst.stride(1)});

    m1_any_md = memory::desc({m, k}, m1_dt, memory::format_tag::any);
    m2_any_md = memory::desc({k, n}, m2_dt, memory::format_tag::any);
    dst_any_md = memory::desc({m, n}, dst_dt, memory::format_tag::any);
  } else {
    m1_md = memory::desc({mb, m, k}, m1_dt, {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_md = attr.m2_trans_ ?
            memory::desc({mb, k, n}, m2_dt, {m2.stride(0), m2.stride(1), m2.stride(2)}) :
            memory::desc({mb, k, n}, m2_dt, {m2.stride(0), m2.stride(2), m2.stride(1)});
    dst_md = memory::desc({mb, m, n}, dst_dt, {dst.stride(0), dst.stride(1), dst.stride(2)});

    m1_usr_md = memory::desc({mb, m, k}, m1_usr_dt, {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_usr_md = attr.m2_trans_ ?
                memory::desc({mb, k, n}, m2_usr_dt, {m2.stride(0), m2.stride(1), m2.stride(2)}) :
                memory::desc({mb, k, n}, m2_usr_dt, {m2.stride(0), m2.stride(2), m2.stride(1)});
    dst_usr_md = memory::desc({mb, m, n}, dst_usr_dt, {dst.stride(0), dst.stride(1), dst.stride(2)});
  }

  // STEP2: creat attribute
  primitive_attr pattr;
  post_ops po;
  int64_t post_flags = 0;
  if (attr.alpha_ != 1.f)
    pattr.set_output_scales(/* mask */ 0, {(float)attr.alpha_});
  // Handle difference cases based-on beta value here:
  // 1. beta == 0, nothing is needed to do
  // 2. quantization path, no bias fusion support in oneDNN so far
  // 3. beta == 1, partially support bias fusion in oneDNN
  // 4. alpha != 1, post-sum is needed for, alpha * (m1 x m2) + post
  if (attr.beta_ != 0.f && (attr.alpha_ != 1.f ||
                            attr.beta_ != 1.f ||
                            m1.is_quantized() ||
                            m2.is_quantized())) {
    po.append_sum(attr.beta_);
    post_flags |= xpu::oneDNN::with_sum;
  }

  if (attr.with_relu()) {
    po.append_eltwise(1.f, algorithm::eltwise_relu, 0.f, 0.f);
    post_flags |= xpu::oneDNN::with_relu;
  }

  if (attr.with_sigmoid()) {
    po.append_eltwise(1.f, algorithm::eltwise_logistic, 0.f, 0.f);
    post_flags |= xpu::oneDNN::with_sigmoid;
  }
  pattr.set_post_ops(po);

  std::vector<float> weight_scales;
  if (m2.is_quantized()) {
    if (m2.qscheme() == kPerTensorAffine) {
      weight_scales.push_back(static_cast<float>(m2.q_scale()));
    } else {
      for (int i = 0; i < m2.size(1); i++)
        weight_scales.push_back(m2.q_per_channel_scales()[i].item<float>());
    }
  }

  if (m1.is_quantized()) {
    auto in_scale = m1.q_scale();
    auto out_scale = dst.is_quantized()? dst.q_scale() : 1.f;
    std::vector<float> matmul_scale;
    for(int i=0; i<weight_scales.size(); i++){
      matmul_scale.push_back(1.f / (out_scale / (in_scale * weight_scales[i])));
    }
    int mask_ac = 0;
    int mask_matmul = weight_scales.size() > 1? 1 << 1 : 0;
    pattr.set_output_scales(mask_matmul, matmul_scale);
    pattr.set_zero_points(DNNL_ARG_DST, mask_ac,
        {static_cast<int>(dst.is_quantized()? dst.q_zero_point() : 0)});
  }

  // STEP3: create primitive
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
#endif

  auto matmul_desc = matmul::desc(m1_md, m2_md, dst_md);

  if (attr.beta_ == 1.f && attr.alpha_ == 1.f &&
      (!m1.is_quantized()) && (!m2.is_quantized())) {
    auto b_dt = b.defined() ? get_onednn_dtype(b) : memory::data_type::f32;
    if (b.sizes() != dst.sizes()) {
      memory::dims b_dims(dst.sizes().size() - 1, 1);
      b_dims.push_back(n);
      b_md = memory::desc(b_dims, b_dt, dst.sizes().size() == 2 ?
          memory::format_tag::ab : memory::format_tag::abc);
    } else {
      if (dims == 2)
        b_md = memory::desc({m, n}, b_dt, {b.stride(0), b.stride(1)});
      else
        b_md = memory::desc({mb, m, n}, b_dt, {b.stride(0), b.stride(1), b.stride(2)});
    }

    if (dims == 2 && lazy_reorder_enabled()) {
      // attr + blk
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_any_md, m2_any_md, b_md, dst_any_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc = matmul::desc(m1_any_md, m2_any_md, b_md, dst_any_md);
    } else {
      // attr + plain
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md, m2_md, b_md, dst_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc = matmul::desc(m1_md, m2_md, b_md, dst_md);
    }
  } else {
    if (dims == 2 && lazy_reorder_enabled()) {
      // no attr + blk
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_any_md, m2_any_md, dst_any_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc = matmul::desc(m1_any_md, m2_any_md, dst_any_md);
    } else {
      // no attr + plain
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md, m2_md, dst_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc = matmul::desc(m1_md, m2_md, dst_md);
    }
  }

  auto matmul_pd = matmul::primitive_desc(matmul_desc, pattr, engine);

#ifdef USE_PRIMITIVE_CACHE
  auto matmul_p = fetch_or_create_m<dnnl::matmul>(key, matmul_pd);
#else
  auto matmul_p = dnnl::matmul(matmul_pd);
#endif

  // STEP4: create memory
  auto m1_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m1);
  auto m1_usr_m = m1_ctx.is_plain() ?
      dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr()) :
      dpcpp_onednn_memory({m1_ctx.meta()}, engine, m1.data_ptr());

  auto m2_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m2);
  auto m2_usr_m = m2_ctx.is_plain() ?
      dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr()) :
      dpcpp_onednn_memory({m2_ctx.meta()}, engine, m2.data_ptr());

  auto dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(dst);
  auto dst_usr_m = dst_ctx.is_plain() ?
      dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr()) :
      dpcpp_onednn_memory({dst_ctx.meta()}, engine, dst.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  Tensor m1_, m2_, dst_;

  // reorder cases
  // case1: master weight support to reorder data type
  // case2: block format support to reorder format
  if (m1_usr_m.get_desc() != expected_m1_md) {
    m1_ = empty_opaque_tensor(expected_m1_md, m1.options(), c10::nullopt);
    m1_m = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
    xpu::oneDNN::reorder(m1, m1_);
  }

  if (m2_usr_m.get_desc() != expected_m2_md) {
    m2_ = empty_opaque_tensor(expected_m2_md, m2.options(), c10::nullopt);
    m2_m = dpcpp_onednn_memory(expected_m2_md, engine, m2_.data_ptr());
    xpu::oneDNN::reorder(attr.m2_trans_ ? m2 : m2.t(), m2_);

    if (weight_cache_enabled()) {
      strm.wait();
      auto ctx_ = at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(m2_);
      // assume oneDNN.matmul.weight is the permution of torch.nn.Linear.weight
      ctx_.set_permution({1, 0});
      at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(m2, std::move(ctx_));
    }
  }

  // bias add for gen12hp platform
  if (dst_usr_m.get_desc() != expected_dst_md) {
    dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());
    if (attr.beta_ != 1.f)
      xpu::oneDNN::reorder(dst, dst_);
  }

  if (attr.beta_ == 1.f && attr.alpha_ == 1.f &&
      (!m1.is_quantized()) && (!m2.is_quantized())) {
    auto b_m = dpcpp_onednn_memory(b_md, engine, b.data_ptr());
    DPCPP_ONEDNN_EXEC(matmul_p, strm,
                      {{DNNL_ARG_SRC, m1_m},
                       {DNNL_ARG_WEIGHTS, m2_m},
                       {DNNL_ARG_BIAS, b_m},
                       {DNNL_ARG_DST, dst_m}});
  } else {
    DPCPP_ONEDNN_EXEC(matmul_p, strm,
                      {{DNNL_ARG_SRC, m1_m},
                       {DNNL_ARG_WEIGHTS, m2_m},
                       {DNNL_ARG_DST, dst_m}});
  }

  if (lazy_reorder_enabled() && dst_m != dst_usr_m && dims == 2) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }
}

}}
