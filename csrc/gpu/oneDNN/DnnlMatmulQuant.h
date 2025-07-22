#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <oneDNN/Utils.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "DnnlExt.h"
#include "Utils.h"
#include "aten/operators/act_dynamic_quant.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>

using namespace dnnl;
using namespace torch_ipex::xpu::oneDNN;
using namespace at::AtenIpexTypeXPU;

#ifdef USE_PRIMITIVE_CACHE

namespace torch_ipex::xpu {
namespace oneDNN {

static inline void set_quant_primitive_attr(
    const Tensor& input,
    primitive_attr& pattr,
    const Tensor& scale,
    const Tensor& zp,
    const int64_t group_size,
    const int64_t act_quant_mode,
    const int64_t k) {
  // set scale and zero point for matmul args
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  // set scale and zp for quantized activation
  if (act_quant_mode == static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_M) ||
      act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_M_SYM)) {
    pattr.set_scales(
        DNNL_ARG_SRC,
        /* mask */ (1 << 0) + (1 << 1),
        {1, k},
        get_onednn_dtype(scale));
    pattr.set_zero_points(
        DNNL_ARG_SRC,
        /* mask */ (1 << 0) + (1 << 1),
        {1, k},
        memory::data_type::s32);
  } else if (
      act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR) ||
      act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR_SYM)) {
    pattr.set_scales(
        DNNL_ARG_SRC,
        /* mask */ 0,
        {},
        get_onednn_dtype(scale));
    pattr.set_zero_points(
        DNNL_ARG_SRC,
        /* mask */ 0,
        {},
        memory::data_type::s32);
  }
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
  // when both src and weight are integer types, disallow fp16 as fpmath_mode
  // so that int8 gemm can be used for acceleration
  if (act_quant_mode == static_cast<int64_t>(ActQuantScheme::UNQUANT_A))
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

  if (input.scalar_type() == at::ScalarType::BFloat16) {
    pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
  } else if (input.scalar_type() == at::ScalarType::Half) {
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", input.scalar_type());
  }
}

template <typename F>
static at::Tensor dnnl_matmul_w4a16_common(
    Tensor& result, // dst, [b, m, n]
    const Tensor& mat1_, // src, [b, m, k]
    const Tensor& mat2, // quantized weight, [K/8, N] transpose
    const std::optional<Tensor>& bias,
    const Tensor& scale, // [k/group_size, n]
    const Tensor& zp, // [k/group_size, n/8]
    int64_t act_quant_mode, // unquant for w4a16
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx,
    F pattr,
    Tensor res_flat,
    Tensor res1_flat) {
  TORCH_CHECK(
      act_quant_mode >= static_cast<int64_t>(ActQuantScheme::UNQUANT_A) &&
          act_quant_mode <=
              static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_M_SYM),
      "Unsupported quant mode for activation, expect UNQUANT, PER_TENSOR or PER_TOKEN but got",
      act_quant_mode);
  // For GPTQ with desc_act=True scenario
  auto mat1 = g_idx.has_value() ? mat1_.index_select(-1, g_idx.value()) : mat1_;

  // transpose mat2 if mat2's shape is [N, K/8]
  if (m2_trans) {
    mat2.transpose_(0, 1);
  }

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

  Tensor act_scale, act_zp;
  bool quant_act =
      (act_quant_mode != static_cast<int64_t>(ActQuantScheme::UNQUANT_A));
  bool use_sym_quant = is_sym_quant(act_quant_mode);
  if (act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR) ||
      act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR_SYM)) {
    // per-tensor quantization
    std::tie(mat1, act_scale, act_zp) =
        at::AtenIpexTypeXPU::dynamic_per_tensor_quant(mat1, use_sym_quant);
  } else if (
      act_quant_mode == static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_M) ||
      act_quant_mode ==
          static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_M_SYM)) {
    // per-token quantization
    std::tie(mat1, act_scale, act_zp) =
        at::AtenIpexTypeXPU::dynamic_per_token_quant(mat1, use_sym_quant);
  }

  dnnl::joint_dtypes_t jd;
  if (mat1.scalar_type() == at::ScalarType::Half) {
    jd = dnnl::joint_dtypes_t::f16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::BFloat16) {
    jd = dnnl::joint_dtypes_t::bf16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::Char) {
    jd = dnnl::joint_dtypes_t::s8_int4;
  } else if (mat1.scalar_type() == at::ScalarType::Byte) {
    jd = dnnl::joint_dtypes_t::u8_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", mat1.scalar_type());
  }

  bias_shape_t bias_shape;
  bias_data_type_t bias_dtype;
  if (bias.has_value() && bias.value().defined()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      bias_shape = bias_shape_t::scalar;
    } else if (nuelm == m * n) {
      bias_shape = bias_shape_t::mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      bias_shape = bias_shape_t::n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      bias_shape = bias_shape_t::m;
    } else if (nuelm == 0) {
      bias_shape = bias_shape_t::none;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }

    switch (b.scalar_type()) {
      case at::ScalarType::Float:
        bias_dtype = bias_data_type_t::f32;
        break;
      case at::ScalarType::BFloat16:
        bias_dtype = bias_data_type_t::bf16;
        break;
      case at::ScalarType::Half:
        bias_dtype = bias_data_type_t::f16;
        break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported data type for bias in int4 matmul: ",
            b.scalar_type());
    }
  } else {
    bias_shape = bias_shape_t::none;
    bias_dtype = bias_data_type_t::none;
  }

  auto mat1_flat = mat1.flatten(0, -2);
  bias_type_t b_type = make_bias_type(bias_shape, bias_dtype);

  const int64_t ldb = mat2.strides()[mat2.dim() - 1] * 8; // for int4 matmul
  const int64_t lda = mat1_flat.strides()[mat1_flat.dim() - 2];
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
  if (quant_act) {
    memory m1_sc_m, m1_zp_m;
    if (act_quant_mode ==
            static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR) ||
        act_quant_mode ==
            static_cast<int64_t>(ActQuantScheme::QUANT_A_PER_TENSOR_SYM)) {
      // per-tensor quantization
      auto Q_PER_TENSOR_SC_MD =
          memory::desc({1}, get_onednn_dtype(act_scale), memory::format_tag::x);
      auto Q_PER_TENSOR_ZP_MD =
          memory::desc({1}, get_onednn_dtype(act_zp), memory::format_tag::x);
      m1_sc_m =
          dpcpp_onednn_memory(Q_PER_TENSOR_SC_MD, engine, act_scale.data_ptr());
      m1_zp_m =
          dpcpp_onednn_memory(Q_PER_TENSOR_ZP_MD, engine, act_zp.data_ptr());
    } else {
      // per-token quantization
      auto Q_PER_TOKEN_SC_MD = memory::desc(
          {m, 1}, get_onednn_dtype(act_scale), memory::format_tag::ab);
      auto Q_PER_TOKEN_ZP_MD = memory::desc(
          {m, 1}, get_onednn_dtype(act_zp), memory::format_tag::ab);
      m1_sc_m =
          dpcpp_onednn_memory(Q_PER_TOKEN_SC_MD, engine, act_scale.data_ptr());
      m1_zp_m =
          dpcpp_onednn_memory(Q_PER_TOKEN_ZP_MD, engine, act_zp.data_ptr());
    }

    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
        act_scale.data_ptr(),
        [&]() { return m1_sc_m; });
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
        act_zp.data_ptr(),
        [&]() { return m1_zp_m; });
  }

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
          int k = mat1_flat.sizes()[1];

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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION("dnnl_matmul_w4a16", std::vector<c10::IValue>({mat1, mat2}));

  auto quant = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu", std::vector<c10::IValue>({mat1, mat2}));

  auto silu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_resmul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto resmul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_gelu",
      std::vector<c10::IValue>({mat1, mat2}));

  auto bias_gelu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_resadd_resadd",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto res1_flat = res1.flatten(0, -2);
  auto bias_resadd_resadd = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu_mul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_silu_mul",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
    const c10::optional<Tensor>& g_idx,
    int64_t act_quant_mode = -1) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(
        mat1, pattr, scale, zp, group_size, act_quant_mode, mat1.size(-1));
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
      act_quant_mode,
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
  const int n = o_sz.back(); // presume channel last format
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
  bias_shape_t bias_shape;
  bias_data_type_t bias_dtype;
  if (bias.has_value() && bias.value().defined()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      bias_shape = bias_shape_t::scalar;
    } else if (nuelm == m * n) {
      bias_shape = bias_shape_t::mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      bias_shape = bias_shape_t::n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      bias_shape = bias_shape_t::m;
    } else if (nuelm == 0) {
      bias_shape = bias_shape_t::none;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }

    switch (b.scalar_type()) {
      case at::ScalarType::Float:
        bias_dtype = bias_data_type_t::f32;
        break;
      case at::ScalarType::BFloat16:
        bias_dtype = bias_data_type_t::bf16;
        break;
      case at::ScalarType::Half:
        bias_dtype = bias_data_type_t::f16;
        break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported data type for bias in int4 matmul: ",
            b.scalar_type());
    }
  } else {
    bias_shape = bias_shape_t::none;
    bias_dtype = bias_data_type_t::none;
  }

  bias_type_t b_type = make_bias_type(bias_shape, bias_dtype);

  dnnl::trans_type_t tt = dnnl::trans_type_t::nn;
  if (trans_b) {
    // transpose mat2
    tt = dnnl::trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for fp8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
      ? mat2.strides()[mat2.dim() - 2]
      : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

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
  if (bias_shape != bias_shape_t::none) {
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
