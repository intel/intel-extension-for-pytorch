#pragma once

#include <ATen/ATen.h>

#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {
static inline void matmul(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& b_raw,
    bool m2_trans,
    Attr attr) {
  size_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  Tensor m1 =
      xpu::oneDNN::is_onednn_matmul_strides(mat1) ? mat1 : mat1.contiguous();
  Tensor m2 =
      xpu::oneDNN::is_onednn_matmul_strides(mat2) ? mat2 : mat2.contiguous();
  Tensor dst = xpu::oneDNN::is_onednn_matmul_strides(result, true)
      ? result
      : result.contiguous();

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  if (dims == 3) {
    mb = dst.size(0);
    TORCH_CHECK(
        mb == m1.size(0) && mb == m2.size(0),
        "batch size mismatch, dst mb: ",
        mb,
        "m1 mb",
        m1.size(0),
        " m2 mb: ",
        m2.size(0));
  }

  // validate bias and make it compatible with oneDNN implementation
  bool with_bias = false;
  Tensor b = b_raw;
  if (b.defined()) {
    with_bias = true;
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
  b = b.contiguous(); // avoid reorder 2 times

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
  } else if (
      m1_dt == memory::data_type::f32 && m2_dt == memory::data_type::bf16) {
    m1_dt = memory::data_type::bf16;
    dst_dt = memory::data_type::bf16;
  }

  if (dims == 2) {
    m1_md = memory::desc({m, k}, m1_dt, {m1.stride(0), m1.stride(1)});
    m2_md = m2_trans
        ? memory::desc({k, n}, m2_dt, {m2.stride(0), m2.stride(1)})
        : memory::desc({k, n}, m2_dt, {m2.stride(1), m2.stride(0)});
    dst_md = memory::desc({m, n}, dst_dt, {dst.stride(0), dst.stride(1)});

    m1_usr_md = memory::desc({m, k}, m1_usr_dt, {m1.stride(0), m1.stride(1)});
    m2_usr_md = m2_trans
        ? memory::desc({k, n}, m2_usr_dt, {m2.stride(0), m2.stride(1)})
        : memory::desc({k, n}, m2_usr_dt, {m2.stride(1), m2.stride(0)});
    dst_usr_md =
        memory::desc({m, n}, dst_usr_dt, {dst.stride(0), dst.stride(1)});

    m1_any_md = memory::desc({m, k}, m1_dt, memory::format_tag::any);
    m2_any_md = memory::desc({k, n}, m2_dt, memory::format_tag::any);
    dst_any_md = memory::desc({m, n}, dst_dt, memory::format_tag::any);
  } else {
    m1_md = memory::desc(
        {mb, m, k}, m1_dt, {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_md = m2_trans
        ? memory::desc(
              {mb, k, n}, m2_dt, {m2.stride(0), m2.stride(1), m2.stride(2)})
        : memory::desc(
              {mb, k, n}, m2_dt, {m2.stride(0), m2.stride(2), m2.stride(1)});
    dst_md = memory::desc(
        {mb, m, n}, dst_dt, {dst.stride(0), dst.stride(1), dst.stride(2)});

    m1_usr_md = memory::desc(
        {mb, m, k}, m1_usr_dt, {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_usr_md = m2_trans
        ? memory::desc(
              {mb, k, n}, m2_usr_dt, {m2.stride(0), m2.stride(1), m2.stride(2)})
        : memory::desc(
              {mb, k, n},
              m2_usr_dt,
              {m2.stride(0), m2.stride(2), m2.stride(1)});
    dst_usr_md = memory::desc(
        {mb, m, n}, dst_usr_dt, {dst.stride(0), dst.stride(1), dst.stride(2)});
  }

  // STEP2: creat attribute
  primitive_attr pattr;

#ifdef BUILD_PRIOR_SYMM_QUANT
  // Only setting zp mask when zp is not zero
  // See: [Note: Use symmetric quant implementation when zp is 0]
  bool src_need_zp = m1.is_quantized() && requires_runtime_zp(m1);
  bool dst_need_zp = dst.is_quantized() && requires_runtime_zp(dst);
  bool wgh_need_zp = m2.is_quantized() && requires_runtime_zp(m2);
#endif

  if (m1.is_quantized()) {
    auto in_scale = m1.q_scale();
    int mask_ac = 0;
    // See [Note: Per-channel quantization mask setting]
    // 1<<1 = 2^1, quantize on second channel of weight aka n in [k, n]
    int mask_wgh = (m2.qscheme() == kPerChannelAffine) ? 1 << 1 : 0;
    if (dst.is_quantized())
      pattr.set_scales_mask(DNNL_ARG_DST, mask_ac);
    pattr.set_scales_mask(DNNL_ARG_SRC, mask_ac);
    pattr.set_scales_mask(DNNL_ARG_WEIGHTS, mask_wgh);

#ifdef BUILD_PRIOR_SYMM_QUANT
    // Only setting zp mask when zp is not zero
    // See: [Note: Use symmetric quant implementation when zp is 0]
    if (src_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_DST, mask_ac);
    if (dst_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
    if (wgh_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_WEIGHTS, mask_wgh);
#endif
  }

  std::unordered_map<int, memory> args;
  post_ops po;
  attr.extract_post_ops(po, dst);
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (m1_dt == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

  // STEP3: create primitive
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto is_onednn_layout_suggested = using_onednn_layout_for_matmul(m1);

  matmul::primitive_desc matmul_pd;

  if (with_bias && (!m1.is_quantized()) && (!m2.is_quantized())) {
    // ensure getting a valid oneDNN bias md here
    b_md = memory::desc(
        get_onednn_dims(b), get_onednn_dtype(b), get_onednn_strides(b));

    if (dims == 2 && is_onednn_layout_suggested) {
      // attr + blk
      matmul_pd = matmul::primitive_desc(
          engine, m1_any_md, m2_any_md, b_md, dst_any_md, pattr);
    } else {
      // attr + plain
      matmul_pd =
          matmul::primitive_desc(engine, m1_md, m2_md, b_md, dst_md, pattr);
    }
  } else {
    if (dims == 2 && is_onednn_layout_suggested) {
      // no attr + blk
      matmul_pd = matmul::primitive_desc(
          engine, m1_any_md, m2_any_md, dst_any_md, pattr);
    } else {
      // no attr + plain
      matmul_pd = matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
    }
  }

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, m1.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
#endif

  auto matmul_p = dnnl::matmul(matmul_pd);

  // STEP4: create memory
  auto m1_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m1);
  auto m1_usr_m = m1_ctx.is_plain()
      ? dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr())
      : dpcpp_onednn_memory({m1_ctx.meta()}, engine, m1.data_ptr());

  auto m2_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m2);
  auto m2_usr_m = m2_ctx.is_plain()
      ? dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr())
      : dpcpp_onednn_memory({m2_ctx.meta()}, engine, m2.data_ptr());

  auto dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(dst);
  auto dst_usr_m = dst_ctx.is_plain()
      ? dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr())
      : dpcpp_onednn_memory({dst_ctx.meta()}, engine, dst.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  Tensor m1_, m2_, dst_;

  auto weight_cache_optimization = [&]() {
    bool onoff = false;
    onoff |= is_onednn_layout_suggested;
    onoff &= c10::InferenceMode::is_enabled();
    return onoff;
  }();

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
    auto m2_onednn_matmul_shape_compatible = m2_trans ? m2 : m2.t();
    xpu::oneDNN::reorder(m2_onednn_matmul_shape_compatible, m2_);

    if (weight_cache_optimization) {
      auto ctx_ =
          at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(m2_);
      // assume oneDNN.matmul.weight is the permution of torch.nn.Linear.weight
      ctx_.set_aten_meta(
          {m2_onednn_matmul_shape_compatible.sizes().vec(),
           m2_onednn_matmul_shape_compatible.strides().vec()});
      at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(
          m2, std::move(ctx_));
    }
  }

  // bias add for gen12hp platform
  if (dst_usr_m.get_desc() != expected_dst_md) {
    dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());
    if (attr.with_sum())
      xpu::oneDNN::reorder(dst, dst_);
  }
  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, po, args);

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (b.defined() && (!m1.is_quantized()) && (!m2.is_quantized())) {
    auto b_m = dpcpp_onednn_memory(b_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, b_m});
  }
#ifdef USE_SCRATCHPAD_MODE
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif

  // TODO: Separate quantized path from fp32 path
  if ((!m1.is_quantized()) && (!m2.is_quantized())) {
    // Path1: normal path for non quantized input
    DPCPP_ONEDNN_EXEC(matmul_p, strm, args);
  } else {
    // Path2: quantized path, set runtime sale and zp here
    bool is_per_tensor_quantized = (m2.qscheme() == kPerTensorAffine);

    memory m1_sc_m, m1_zp_m;
    memory::desc m1_sc_md =
        memory::desc({1}, memory::data_type::f32, memory::format_tag::x);
    std::tie(m1_sc_m, m1_zp_m) = q_get_sc_zp_gpu_mem(m1, engine);
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc_m});

    memory dst_sc_m, dst_zp_m;
    if (dst.is_quantized()) {
      std::tie(dst_sc_m, dst_zp_m) = q_get_sc_zp_gpu_mem(dst, engine);
      args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_m});
    }

#ifdef BUILD_PRIOR_SYMM_QUANT
    // Only setting zp when zp is not zero
    // See: [Note: Use symmetric quant implementation when zp is 0]
    if (src_need_zp) {
      args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, m1_zp_m});
    }
#endif

#ifdef BUILD_PRIOR_SYMM_QUANT
    // Only setting zp when zp is not zero
    // See: [Note: Use symmetric quant implementation when zp is 0]
    if (dst.is_quantized() && dst_need_zp) {
      args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_m});
    }
#endif

    if (is_per_tensor_quantized) {
      memory m2_sc_m;
      m2_sc_m = q_get_wgh_sc_gpu_mem(m2, engine);
      args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m2_sc_m});

      DPCPP_ONEDNN_EXEC(matmul_p, strm, args);
    } else {
      // Per-channel quantized
      Tensor wgh_sc = m2.q_per_channel_scales();
      memory::desc wgh_sc_md = memory::desc(
          get_onednn_dims(wgh_sc),
          memory::data_type::f32,
          memory::format_tag::x);
      memory wgh_sc_m =
          dpcpp_onednn_memory(wgh_sc_md, engine, wgh_sc.data_ptr());
      args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wgh_sc_m});
      DPCPP_ONEDNN_EXEC(matmul_p, strm, args);
    }
  }
  if (is_onednn_layout_suggested && dst_m != dst_usr_m && dims == 2) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  if (!dst.is_same(result))
    result.copy_(dst);
}

} // namespace oneDNN
} // namespace xpu
