#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "DnnlExt.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>

using namespace dnnl;
using namespace torch_ipex::xpu::oneDNN;

#ifdef USE_PRIMITIVE_CACHE

namespace torch_ipex::xpu {
namespace oneDNN {

static inline void set_quant_primitive_attr(
    primitive_attr& pattr,
    const Tensor& scale,
    const Tensor& zp,
    const int64_t group_size) {
  // set scale and zero point for matmul args
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      get_onednn_dtype(scale));

  if (zp.dim() == 1) {
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ 0,
        {},
        memory::data_type::s8);
  } else {
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        memory::data_type::u4);
  }
  pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
}

template <typename F>
static at::Tensor dnnl_matmul_w4a16_common(
    Tensor& result, // dst, [b, m, n]
    const Tensor& mat1_, // src, [b, m, k]
    const Tensor& mat2, // quantized weight, [K/8, N] transpose
    const std::optional<Tensor>& bias,
    const Tensor& scale, // [k/group_size, n]
    const Tensor& zp, // [k/group_size, n/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx,
    F pattr,
    Tensor res_flat,
    Tensor res1_flat) {
  // For GPTQ with desc_act=True scenario
  auto mat1 = g_idx.has_value() ? mat1_.index_select(-1, g_idx.value()) : mat1_;

  auto src_sz = mat1.sizes();
  auto o_sz = mat1.sizes().vec();
  auto b_sz = mat2.sizes();
  *(o_sz.end() - 1) = *(b_sz.end() - 1);
  result = at::empty(o_sz, mat1.options());

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = b_sz[1]; // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get device, engine, stream
  const int device_id = at::xpu::current_device();
  at::Device curDevice = at::Device(at::kXPU, device_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  dnnl::joint_dtypes_t jd;
  if (mat1.scalar_type() == at::ScalarType::Half) {
    jd = dnnl::joint_dtypes_t::f16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::BFloat16) {
    jd = dnnl::joint_dtypes_t::bf16_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", mat1.scalar_type());
  }

  bias_type_t b_type;
  if (bias.has_value() && bias.value().defined()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      b_type = bias_type_t::scalar;
    } else if (nuelm == m * n) {
      b_type = bias_type_t::mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      b_type = bias_type_t::n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      b_type = bias_type_t::m;
    } else if (nuelm == 0) {
      b_type = bias_type_t::none;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }
  } else {
    b_type = bias_type_t::none;
  }

  const int64_t ldb = mat2.strides()[mat2.dim() - 1] * 8; // for int4 matmul
  const int64_t lda = mat1.strides()[mat1.dim() - 2];
  const int64_t ldc = result.strides()[result.dim() - 2];

  // only support nt for int4 matmul
  trans_type_t tt = trans_type_t::nt;
  int64_t zp_group_size = zp.dim() == 1 ? 1 : group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      tt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      device_id,
      pattr,
      group_size,
      zp_group_size);

  int arg_off = 0;
  // set scale and zero point for matmul args
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      scale.data_ptr(),
      [&]() {
        return dpcpp_onednn_memory(
            get_onednn_md(scale), engine, scale.data_ptr());
      });

  // set zp_md for symmetric quantization
  if (zp.dim() == 1) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        zp.data_ptr(),
        [&]() {
          return dpcpp_onednn_memory(get_onednn_md(zp), engine, zp.data_ptr());
        });
  } else {
    // set zp_md for asymmetric quantization
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        zp.data_ptr(),
        [&]() {
          int n = mat2.sizes()[1];
          int k = mat1.sizes()[1];

          const uint64_t num_groups = (uint64_t)(k / group_size);
          memory zp_B_u4_m(
              {{num_groups, n}, memory::data_type::u4, {n, 1}},
              engine,
              zp.data_ptr());
          return zp_B_u4_m;
        });
  }

  if (res_flat.defined()) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
        res_flat.data_ptr(),
        [&]() {
          return dpcpp_onednn_memory(
              get_onednn_md(res_flat), engine, res_flat.data_ptr());
        });
  }
  if (res1_flat.defined()) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
        res1_flat.data_ptr(),
        [&]() {
          return dpcpp_onednn_memory(
              get_onednn_md(res1_flat), engine, res1_flat.data_ptr());
        });
  }

  // set general args
  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (bias.has_value()) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());
#endif

  auto strm = GpuStreamManager::Instance().get_stream();
  /* matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off); */
  DPCPP_ONEDNN_EXEC_WITH_ARGHANDLES(
      matmul_ext, strm, engine, arg_handles, arg_off);

  return result;
}

static at::Tensor dnnl_matmul_w4a16(
    Tensor& result, // dst, [b, m, n]
    const Tensor& mat1, // src, [b, m, k]
    const Tensor& mat2, // quantized weight, [k, n] transpose
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [k/group_size, n]
    const Tensor& zp, // [k/group_size, n/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION("dnnl_matmul_w4a16", std::vector<c10::IValue>({mat1, mat2}));

  auto quant = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      quant,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_silu(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu", std::vector<c10::IValue>({mat1, mat2}));

  auto silu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_resmul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_resmul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto resmul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      resmul,
      res_flat,
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_gelu(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    c10::string_view approximate,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_gelu",
      std::vector<c10::IValue>({mat1, mat2}));

  auto bias_gelu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    if (approximate == "none") {
      po.append_eltwise(algorithm::eltwise_gelu_erf, 1.f, 0.f);
    } else if (approximate == "tanh") {
      po.append_eltwise(algorithm::eltwise_gelu_tanh, 1.f, 0.f);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
    }
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_gelu,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_resadd_resadd(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    const Tensor& res1,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_resadd_resadd",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto res1_flat = res1.flatten(0, -2);
  auto bias_resadd_resadd = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    po.append_binary(algorithm::binary_add, get_onednn_md(res1_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_resadd_resadd,
      res_flat,
      res1_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_silu_mul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu_mul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  auto matmul_ext = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu_mul,
      at::Tensor(),
      res_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_silu_mul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_silu_mul",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  auto matmul_ext = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu_mul_int4,
      at::Tensor(),
      res_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_add(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_add_int4,
      res_flat,
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_add(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_add_int4,
      res_flat,
      at::Tensor());

  return result;
}

static inline void dnnl_matmul_w8a16_fp8(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2,
    bool trans_b,
    const std::optional<Tensor>& bias,
    const Tensor& m2_sc,
    const int64_t group_size = 0) {
  TORCH_CHECK(
      mat2.scalar_type() == at::ScalarType::Float8_e5m2 ||
          mat2.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();
  // auto b_sz = mat2.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz[1]; // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get device, engine, stream
  const int device_id = at::xpu::current_device();
  at::Device curDevice = at::Device(at::kXPU, device_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  // get joint dtypes
  dnnl::joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  if (in_dtype == at::ScalarType::Half) {
    jd = wei_dtype == at::ScalarType::Float8_e5m2
        ? dnnl::joint_dtypes_t::f16_f8_e5m2
        : dnnl::joint_dtypes_t::f16_f8_e4m3;
  } else if (in_dtype == at::ScalarType::BFloat16) {
    jd = wei_dtype == at::ScalarType::Float8_e5m2
        ? dnnl::joint_dtypes_t::bf16_f8_e5m2
        : dnnl::joint_dtypes_t::bf16_f8_e4m3;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for fp8 matmul: ", mat1.scalar_type());
  }

  // get bias type
  bias_type_t b_type;
  if (bias.has_value() && bias.value().defined()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      b_type = bias_type_t::scalar;
    } else if (nuelm == m * n) {
      b_type = bias_type_t::mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      b_type = bias_type_t::n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      b_type = bias_type_t::m;
    } else if (nuelm == 0) {
      b_type = bias_type_t::none;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }
  } else {
    b_type = bias_type_t::none;
  }

  dnnl::trans_type_t tt = dnnl::trans_type_t::nn;
  if (trans_b) {
    // transpose mat2
    tt = dnnl::trans_type_t::nt;
  }

  int64_t lda = mat1.strides()[mat1.dim() - 2];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
      ? mat2.strides()[mat2.dim() - 2]
      : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[result.dim() - 2];

  auto f_attr = [&](primitive_attr& pattr) {
#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
    // only for per-tensor scaling
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ 0,
        {},
        get_onednn_dtype(m2_sc));
  };

  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, device_id, f_attr, group_size);

  int arg_off = 0;

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return dpcpp_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (b_type != bias_type_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());
#endif

  auto strm = GpuStreamManager::Instance().get_stream();
  DPCPP_ONEDNN_EXEC_WITH_ARGHANDLES(
      matmul_ext, strm, engine, arg_handles, arg_off);
}

} // namespace oneDNN
} // namespace torch_ipex::xpu

#endif // USE_PRIMITIVE_CACHE
