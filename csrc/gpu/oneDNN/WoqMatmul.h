#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include "operators/BlasImpl.h"

using namespace dnnl;
using namespace torch_ipex::xpu::oneDNN;

#undef USE_PRIMITIVE_CACHE

namespace torch_ipex::xpu {
namespace oneDNN {

#define RECORD_WOQ_FUNCTION_IMPL(F, m, n, k, group_size, m2_trans) \
  char str__[100];                                                 \
  sprintf(                                                         \
      str__,                                                       \
      "woq_%s(%d, %d, %d, s=%d, wt=%d)",                           \
      "" #F,                                                       \
      m,                                                           \
      n,                                                           \
      k,                                                           \
      group_size,                                                  \
      m2_trans);                                                   \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

inline Tensor resize_as_onednn_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

static Tensor woq_matmul_int4(
    Tensor& result, // dst, [M, N]
    const Tensor& mat1_, // src, [M, K]
    const Tensor& mat2_, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  Tensor mat1;
  if (g_idx.has_value()) {
    mat1 = mat1_.index_select(-1, g_idx.value()).flatten(0, -2);
  } else {
    mat1 = mat1_.flatten(0, -2);
  }
  auto mat2 = mat2_.flatten(0, -2);
  int m = mat1.sizes()[0];
  int n = mat2.sizes()[1];
  int k = mat1.sizes()[1];
  RECORD_WOQ_FUNCTION_IMPL(matmul_int4, m, n, k, group_size, m2_trans);
  result = at::empty({m, n}, mat1_.options());
  size_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  // get device, engine, stream
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // engine index means the engine created on which device
  auto engine_index = curDevice.index();
  auto strm = GpuStreamManager::Instance().get_stream();

  // make them all contiguous
  Tensor m1 = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(mat1)
      ? mat1
      : contiguous_if_needed(mat1);
  Tensor m2 = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(mat2)
      ? mat2
      : contiguous_if_needed(mat2);
  Tensor scale_ = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(scale)
      ? scale
      : contiguous_if_needed(scale);
  Tensor zp_ = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(zp)
      ? zp
      : contiguous_if_needed(zp);
  Tensor dst = torch_ipex::xpu::oneDNN::is_onednn_matmul_strides(result, true)
      ? result
      : contiguous_if_needed(result);

  m = dst.size(-2);
  n = dst.size(-1);
  k = m1.size(-1);
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

  // bias is fused in post-op for quantized path
  b = b.contiguous(); // avoid reorder 2 times

  // ipex matmul support both ab/ba shape for m2 tensor, we don't check any more
  // convert torch dtype to oneDNN defined dtype
  // torch tensor of int32 dtype, corresponding to oneDNN s32
  auto m1_usr_dt = get_onednn_dtype(m1); // half <==> f16
  //   auto m2_usr_dt = memory::data_type::u4; // int32, representing 8xint4
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto scale_user_dt = get_onednn_dtype(scale_); // half <==> fp16
  //   auto zp_user_dt = memory::data_type::u4; // int32, representing 8xint4
  auto zp_user_dt = get_onednn_dtype(zp_);
  auto dst_usr_dt = get_onednn_dtype(dst); // half <==> f16

  auto m1_dt = m1_usr_dt;
  auto m2_dt = memory::data_type::u4;
  auto scale_dt = scale_user_dt;
  auto zp_dt = memory::data_type::u4;
  auto dst_dt = dst_usr_dt;
  memory::data_type bias_dt;

  memory::desc m1_md, m1_usr_md;
  memory::desc m2_md, m2_usr_md;
  memory::desc scale_md, scale_usr_md;
  memory::desc zp_md, zp_usr_md;
  memory::desc dst_md, dst_usr_md;
  memory::desc b_md;

  // STEP1: create memory desc
  memory::dims m1_dims, m2_dims, m2_usr_dims, scale_dims, zp_dims, zp_usr_dims,
      dst_dims, bias_dims;
  memory::dims m1_strides, m2_strides, m2_usr_strides, scale_strides,
      zp_strides, zp_usr_strides, dst_strides, bias_strides;

  const uint64_t num_groups = (uint64_t)(k / group_size);
  const uint64_t compressed_n = (uint64_t)(n / 8);
  const uint64_t compressed_k = (uint64_t)(k / 8);

  m2_usr_dims = {compressed_k, n};
  scale_dims = {num_groups, n};
  if (zp.dim() == 1) {
    zp_dims = {1};
    zp_usr_dims = {1};
  } else {
    zp_dims = {num_groups, compressed_n};
    zp_usr_dims = {num_groups, compressed_n};
  }

  m2_usr_strides = {1, compressed_k};
  scale_strides = {scale_.stride(0), scale_.stride(1)};
  if (zp.dim() == 1) {
    zp_strides = {1};
    zp_usr_strides = {1};
  } else {
    zp_strides = {zp.stride(0), zp.stride(1)};
    zp_usr_strides = {zp.stride(0), zp.stride(1)};
  }

  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    m2_strides = {n, 1};
    dst_strides = {dst.stride(0), dst.stride(1)};
  } else {
    m1_dims = {mb, m, k};
    m2_dims = {mb, k, n};
    dst_dims = {mb, m, n};

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1), m2.stride(2)};
    } else {
      m2_strides = {m2.stride(0), m2.stride(2), m2.stride(1)};
    }
    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  if (with_bias) {
    bias_dims = get_onednn_dims(b);
    bias_dt = get_onednn_dtype(b);
    bias_strides = get_onednn_strides(b);
  }

  std::unordered_map<int, memory> args;

  post_ops po;
  attr.extract_post_ops(po, dst);

  auto is_onednn_layout_suggested = using_onednn_layout_for_matmul(m1);

  bool load_from_cache = false;

  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  m1_usr_md = memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = memory::desc(m2_usr_dims, m2_usr_dt, m2_usr_strides);
  scale_usr_md = memory::desc(scale_dims, scale_user_dt, scale_strides);
  zp_usr_md = memory::desc(zp_usr_dims, zp_user_dt, zp_usr_strides);
  dst_usr_md = memory::desc(dst_dims, dst_usr_dt, dst_strides);
  // STEP4: create memory
  auto m1_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m1);
  auto m1_usr_m = m1_ctx.is_plain()
      ? dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr())
      : dpcpp_onednn_memory({m1_ctx.meta()}, engine, m1.data_ptr());

  auto m2_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m2);
  auto m2_usr_m = m2_ctx.is_plain()
      ? dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr())
      : dpcpp_onednn_memory({m2_ctx.meta()}, engine, m2.data_ptr());

  void* handle_b = m2_usr_m.get_data_handle();
  memory m2_u4_m(
      {{k, n}, memory::data_type::u4, memory::format_tag::ba},
      engine,
      handle_b);

  if (is_onednn_layout_suggested && dims == 2) {
    m1_md = memory::desc(m1_dims, m1_dt, memory::format_tag::any);
    m2_md = memory::desc(m2_dims, m2_dt, memory::format_tag::any);
    scale_md = memory::desc(scale_dims, scale_dt, memory::format_tag::any);
    zp_md = memory::desc(zp_dims, zp_dt, memory::format_tag::any);
    dst_md = memory::desc(dst_dims, dst_dt, memory::format_tag::any);
  } else {
    m1_md = memory::desc(m1_dims, m1_dt, m1_strides);
    m2_md = memory::desc(m2_dims, m2_dt, m2_strides);
    scale_md = memory::desc(scale_dims, scale_dt, scale_strides);
    zp_md = memory::desc(zp_dims, zp_dt, zp_strides);
    dst_md = memory::desc(dst_dims, dst_dt, dst_strides);
  }

  auto dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(dst);
  auto dst_usr_m = dst_ctx.is_plain()
      ? dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr())
      : dpcpp_onednn_memory({dst_ctx.meta()}, engine, dst.data_ptr());

  auto scale_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(scale_);
  auto scale_usr_m = scale_ctx.is_plain()
      ? dpcpp_onednn_memory(scale_usr_md, engine, scale.data_ptr())
      : dpcpp_onednn_memory({scale_ctx.meta()}, engine, scale.data_ptr());

  auto zp_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(zp_);
  auto zp_usr_m = zp_ctx.is_plain()
      ? dpcpp_onednn_memory(zp_usr_md, engine, zp.data_ptr())
      : dpcpp_onednn_memory({zp_ctx.meta()}, engine, zp.data_ptr());

  // STEP2: creat attribute
  primitive_attr pattr;
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  // Set scales with multiple scales along K dimension and with groups along
  // K.
  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      scale_dt);
  // Set a single zero point with s8 data type.
  if (zp.dim() == 1) {
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ 0,
        {},
        memory::data_type::s8);
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_usr_m});
  } else {
    void* handle_zp = zp_usr_m.get_data_handle();
    memory zp_B_u4_m(
        {{num_groups, n}, memory::data_type::u4, {n, 1}}, engine, handle_zp);
    // Set zero points with u4 data type.
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        /* groups */ {group_size, 1},
        memory::data_type::u4);
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_B_u4_m});
  }
  // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
  // integral primitives (in this example, matmul).
  // pattr.set_fpmath_mode(
  //     torch_ipex::xpu::oneDNN::get_onednn_fpmath_mode(), true);
  pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

  if (with_bias) {
    b_md = memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd = matmul::primitive_desc(
        engine, m1_md, m2_u4_m.get_desc(), b_md, dst_md, pattr);
  } else {
    matmul_pd = matmul::primitive_desc(
        engine, m1_md, m2_u4_m.get_desc(), dst_md, pattr);
  }

  matmul_p = dnnl::matmul(matmul_pd);

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  memory m1_m = m1_usr_m, m2_m = m2_u4_m, dst_m = dst_usr_m;
  memory scale_m = scale_usr_m; // zp_m = zp_u4_m;
  Tensor m1_, m2_, zp_new, dst_;

  auto weight_cache_optimization = [&]() {
    bool onoff = false;
    onoff |= is_onednn_layout_suggested;
    onoff &= c10::InferenceMode::is_enabled();
    return onoff;
  }();

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, m1.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif

  // reorder cases
  // case1: master weight support to reorder data type
  // case2: block format support to reorder format
  if (m1_usr_m.get_desc() != expected_m1_md) {
    m1_ = empty_opaque_tensor(expected_m1_md, m1.options(), c10::nullopt);
    m1_m = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
    torch_ipex::xpu::oneDNN::reorder(m1, m1_);
  }

  // bias add for gen12hp platform
  if (dst_usr_m.get_desc() != expected_dst_md) {
    dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());
    if (attr.with_sum())
      torch_ipex::xpu::oneDNN::reorder(dst, dst_);
  }
  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_u4_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (b.defined()) {
    auto b_m = dpcpp_onednn_memory(b_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, b_m});
  }
  // add scale & zp
  // memory zp_m({{1}, memory::data_type::s8, {1}}, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_m});

  DPCPP_ONEDNN_EXEC(matmul_p, strm, args);
  if (is_onednn_layout_suggested && dst_m != dst_usr_m && dims == 2) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  if (!dst.is_same(result))
    result.copy_(dst);
  result = resize_as_onednn_mat1(mat1_, result);
  return result;
}

static Tensor& woq_matmul_fusion_variants(
    Tensor& output, // dst, [M, N]
    const Tensor& tensor1_, // src, [M, K]
    const Tensor& tensor2_, // quantized weight, [K/8, N]
    const Tensor& weight_scl, // [K/group_size, N]
    const Tensor& weight_zp, // [k/group_size, N/8]
    int64_t group_size,
    bool trans,
    Attr& attr,
    const c10::optional<Tensor>& g_idx,
    bool& is_fused,
    Tensor bias = at::Tensor()) {
  auto tensor1 = tensor1_.flatten(0, -2);
  auto tensor2 = tensor2_.flatten(0, -2);
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();
  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(
      dim_tensor1 >= 2 && dim_tensor2 >= 2,
      "both arguments to matmul need to be at least 2D, but they are ",
      dim_tensor1,
      "D and ",
      dim_tensor2,
      "D");

  if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // original sizes: [4, 2] x [2, 6] -> [4, 6]
    // onednn sizes: [4, 2] x [2, 6] -> [4, 6]
    DimVector output_shape({tensor1.size(0), tensor2.size(1)});
    if (trans)
      output_shape[1] = tensor2.size(0);

    Tensor result =
        output.defined() ? output : at::empty(output_shape, tensor1.options());

    is_fused = at::AtenIpexTypeXPU::impl::get_onednn_matmul_binary_attr(
        result, attr, dim_tensor1, dim_tensor2, output_shape);
    result = woq_matmul_int4(
        result,
        tensor1_,
        tensor2_,
        weight_scl,
        weight_zp,
        group_size,
        false,
        attr,
        g_idx,
        bias);
    if (output.defined() && !output.is_alias_of(result)) {
      output.copy_(result);
    } else {
      output = result;
    }
    return output;
  } else {
    TORCH_CHECK(
        false,
        "matmul with tensors of dim ",
        dim_tensor1,
        " and ",
        dim_tensor2,
        " is not supported");
  }
}

at::Tensor woq_matmul_silu(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_silu", std::vector<c10::IValue>({tensor1, tensor2}));
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_swish);
  bool is_fused;
  return woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused);
}

at::Tensor woq_matmul_resmul(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_resmul", std::vector<c10::IValue>({tensor1, tensor2}));
  auto res_flat = res.flatten(0, -2);
  attr.append_post_binary(attr.kind_with_binary_mul, res_flat);
  bool is_fused;
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused);
  if (!is_fused) {
    result = result * res;
  }
  return result;
}

at::Tensor woq_matmul_bias_gelu(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    c10::string_view approximate,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_bias_gelu", std::vector<c10::IValue>({tensor1, tensor2}));
  algorithm algo;
  if (approximate == "none") {
    algo = attr.kind_with_gelu_erf;
  } else if (approximate == "tanh") {
    algo = attr.kind_with_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
  }
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      algo);
  bool is_fused;
  return woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused,
      b_raw);
}

at::Tensor woq_matmul_bias_resadd_resadd(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res0,
    const Tensor& res1,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor bias = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_bias_resadd_resadd",
      std::vector<c10::IValue>({tensor1, tensor2}));
  auto res0_flat = res0.flatten(0, -2);
  auto res1_flat = res1.flatten(0, -2);
  bool is_fused;
  attr.append_post_binary(attr.kind_with_binary_add, bias);
  attr.append_post_binary(attr.kind_with_binary_add, res0_flat);
  attr.append_post_binary(attr.kind_with_binary_add, res1_flat);
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused);
  if (!is_fused) {
    result += bias + res0 + res1;
  }
  return result;
}

at::Tensor woq_matmul_silu_mul(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_silu_mul", std::vector<c10::IValue>({tensor1, tensor2}));
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_swish);
  auto res_flat = res.flatten(0, -2);
  attr.append_post_binary(attr.kind_with_binary_mul, res_flat);
  bool is_fused;
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused);
  return result;
}

at::Tensor woq_matmul_bias_silu_mul_int4(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor bias = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_bias_silu_mul_int4",
      std::vector<c10::IValue>({tensor1, tensor2}));
  attr.append_post_eltwise(
      /* scale */ 1.f,
      /* alpha */ 1.f,
      /* beta */ 0.f,
      attr.kind_with_swish);
  auto res_flat = res.flatten(0, -2);
  attr.append_post_binary(attr.kind_with_binary_mul, res_flat);
  bool is_fused;
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused,
      bias);
  return result;
}

at::Tensor woq_matmul_add_int4(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor b_raw = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_add_int4", std::vector<c10::IValue>({tensor1, tensor2}));
  auto res_flat = res.flatten(0, -2);
  attr.append_post_binary(attr.kind_with_binary_add, res_flat);
  bool is_fused;
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused);
  if (!is_fused) {
    result += res;
  }
  return result;
}

at::Tensor woq_matmul_bias_add_int4(
    Tensor& result, // dst, [M, N]
    const Tensor& tensor1, // src, [M, K]
    const Tensor& tensor2, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    Tensor bias = at::Tensor()) {
  RECORD_FUNCTION(
      "woq_matmul_bias_add_int4", std::vector<c10::IValue>({tensor1, tensor2}));
  auto res_flat = res.flatten(0, -2);
  attr.append_post_binary(attr.kind_with_binary_add, res_flat);
  bool is_fused;
  result = woq_matmul_fusion_variants(
      result,
      tensor1,
      tensor2,
      scale,
      zp,
      group_size,
      m2_trans,
      attr,
      g_idx,
      is_fused,
      bias);
  if (!is_fused) {
    result += bias + res;
  }
  return result;
}

} // namespace oneDNN
} // namespace torch_ipex::xpu
