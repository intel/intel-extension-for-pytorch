#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/record_function.h>
#include <ATen/CPUApplyUtils.h>
#include <core/TensorImplUtils.h>
#include <utils/ATDispatch.h>

#include <core/Runtime.h>
#include <oneDNN/oneDNN.h>
#include <dnnl.hpp>
#include <vector>

#include "QUtil.h"

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif
#include <c10/util/typeid.h>

using namespace dnnl;
using namespace at::dpcpp;

namespace caffe2 {
CAFFE_KNOWN_TYPE(at::AtenIpexTypeQuantizedXPU::PackedLinearWeightQDPCPP)
}

namespace at {
namespace impl {

typedef struct matmul_attr {
  static const int64_t kind_with_relu = at::dpcpp::oneDNN::with_relu;
  static const int64_t kind_with_sigmoid = at::dpcpp::oneDNN::with_sigmoid;

  matmul_attr() : alpha_(1.f), beta_(0.f), attr_(0), m2_trans_(true) {}
  matmul_attr(float alpha, float beta, int64_t attr, bool m2_trans) :
      alpha_(alpha), beta_(beta), attr_(attr), m2_trans_(m2_trans) {}

  bool with_relu() {
    return attr_ & kind_with_relu;
  }

  bool with_sigmoid() {
    return attr_ & kind_with_sigmoid;
  }

  int64_t attr() {
    return attr_;
  }

  float alpha_;
  float beta_;
  int64_t attr_;
  bool m2_trans_;
} matmul_attr_t;

void dnnlGemmImpl(
    Tensor& result,
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& b,
    matmul_attr_t attr) {
  size_t dims = result.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "oneDNN matmul only works with 2D or 3D, got ", dims);
  TORCH_CHECK(dims == m1.dim() && dims == m2.dim(), "oneDNN input matrixes must have the same ranks");

  int64_t m = result.size(-2);
  int64_t n = result.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  if (dims == 3) {
    mb = result.size(0);
    TORCH_CHECK(mb == m1.size(0) && mb == m2.size(0), "batch size mismatch, result mb: ",\
        mb, "m1 mb", m1.size(0), " m2 mb: ", m2.size(0));
  }
  // ipex matmul support both ab/ba shape for m2 tensor, we don't check any more

  auto m1_dt = dt_to_dnnl(m1.scalar_type());
  auto m2_dt = dt_to_dnnl(m2.scalar_type());
  auto result_dt = dt_to_dnnl(result.scalar_type());

  auto m1_usr_dt = m1_dt;
  auto m2_usr_dt = m2_dt;
  auto result_usr_dt = result_dt;

  memory::desc m1_md, m1_usr_md;
  memory::desc m2_md, m2_usr_md;
  memory::desc r_md, r_usr_md;
  memory::desc b_md;

  memory::desc m1_md_any;
  memory::desc m2_md_any;
  memory::desc r_md_any;

  // Naive Master weight
  if (m1_dt == dnnl::memory::data_type::bf16 && m2_dt == dnnl::memory::data_type::f32) {
    m2_dt = dnnl::memory::data_type::bf16;
    result_dt = dnnl::memory::data_type::bf16;

  } else if (m1_dt == dnnl::memory::data_type::f32 && m2_dt == dnnl::memory::data_type::bf16) {
    m1_dt = dnnl::memory::data_type::bf16;
    result_dt = dnnl::memory::data_type::bf16;
  }

  if (dims == 2) {
    m1_md = memory::desc({m, k}, m1_dt, {m1.stride(0), m1.stride(1)});
    m2_md = attr.m2_trans_ ? memory::desc({k, n}, m2_dt, {m2.stride(0), m2.stride(1)}) :
            memory::desc({k, n}, m2_dt, {m2.stride(1), m2.stride(0)});
    r_md = memory::desc({m, n}, result_dt, {result.stride(0), result.stride(1)});

    m1_usr_md = memory::desc({m, k}, m1_usr_dt, {m1.stride(0), m1.stride(1)});
    m2_usr_md = attr.m2_trans_ ? memory::desc({k, n}, m2_usr_dt, {m2.stride(0), m2.stride(1)}) :
            memory::desc({k, n}, m2_usr_dt, {m2.stride(1), m2.stride(0)});
    r_usr_md = memory::desc({m, n}, result_usr_dt, {result.stride(0), result.stride(1)});

    m1_md_any = memory::desc({m, k}, m1_dt, memory::format_tag::any);
    m2_md_any = memory::desc({k, n}, m2_dt, memory::format_tag::any);
    r_md_any = memory::desc({m, n}, result_dt, memory::format_tag::any);
  } else {
    m1_md = memory::desc({mb, m, k}, m1_dt,
      {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_md = attr.m2_trans_ ? memory::desc({mb, k, n}, m2_dt,
                                 {m2.stride(0), m2.stride(1), m2.stride(2)}) :
                             memory::desc({mb, k, n}, m2_dt,
                                 {m2.stride(0), m2.stride(2), m2.stride(1)});
    r_md = memory::desc({mb, m, n}, result_dt,
      {result.stride(0), result.stride(1), result.stride(2)});

    m1_usr_md = memory::desc({mb, m, k}, m1_usr_dt,
      {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_usr_md = attr.m2_trans_ ? memory::desc({mb, k, n}, m2_usr_dt,
                                 {m2.stride(0), m2.stride(1), m2.stride(2)}) :
                             memory::desc({mb, k, n}, m2_usr_dt,
                                 {m2.stride(0), m2.stride(2), m2.stride(1)});
    r_usr_md = memory::desc({mb, m, n}, result_usr_dt,
      {result.stride(0), result.stride(1), result.stride(2)});
  }

  primitive_attr pattr;
  post_ops po;
  int64_t post_flags = 0;
  if (attr.alpha_ != 1.f)
    pattr.set_output_scales(/* mask */ 0, {(float)attr.alpha_});
#ifdef USE_GEN12HP_ONEDNN
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
#else
  if (attr.beta_ != 0.f) {
    po.append_sum(attr.beta_);
#endif
    post_flags |= at::dpcpp::oneDNN::with_sum;
  }

  if (attr.with_relu()) {
    po.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    post_flags |= at::dpcpp::oneDNN::with_relu;
  }

  if (attr.with_sigmoid()) {
    po.append_eltwise(1.f, dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    post_flags |= at::dpcpp::oneDNN::with_sigmoid;
  }
  pattr.set_post_ops(po);

  std::vector<float> weight_scales;
  if(m2.is_quantized()){
    if (m2.qscheme() == kPerTensorAffine) {
      weight_scales.push_back(static_cast<float>(m2.q_scale()));
    } else {
      for (int i = 0; i < m2.size(1); i++) {
        weight_scales.push_back(m2.q_per_channel_scales()[i].item<float>());
      }
    }
  }

  if(m1.is_quantized()){
    auto in_scale = m1.q_scale();
    auto out_scale = result.is_quantized()? result.q_scale() : 1.f;
    std::vector<float> matmul_scale;
    for(int i=0; i<weight_scales.size(); i++){
      matmul_scale.push_back(1.f / (out_scale / (in_scale * weight_scales[i])));
    }
    int mask_ac = 0;
    int mask_matmul = weight_scales.size() > 1? 1 << 1 : 0;
    pattr.set_output_scales(mask_matmul, matmul_scale);
    pattr.set_zero_points(DNNL_ARG_DST, mask_ac,
        {static_cast<int>(result.is_quantized()? result.q_zero_point() : 0)});
  }

  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key, key_r;
#endif

  std::shared_ptr<dnnl::matmul::desc> matmul_desc;
#ifdef USE_GEN12HP_ONEDNN
  if (attr.beta_ == 1.f && attr.alpha_ == 1.f && (!m1.is_quantized()) && (!m2.is_quantized())) {
    auto b_dt = b.defined() ? dt_to_dnnl(b.scalar_type()) : dnnl::memory::data_type::f32;
    if (b.sizes() != result.sizes()) {
      dnnl::memory::dims b_dims(result.sizes().size() - 1, 1);
      b_dims.push_back(n);
      b_md = memory::desc(b_dims, b_dt, result.sizes().size() == 2
          ? dnnl::memory::format_tag::ab : dnnl::memory::format_tag::abc);
    } else {
      if (dims == 2) {
        b_md = memory::desc({m, n}, b_dt, {b.stride(0), b.stride(1)});
      } else {
        b_md = memory::desc({mb, m, n}, b_dt, {b.stride(0), b.stride(1), b.stride(2)});
      }
    }
    if (dims == 2 && lazy_reorder_enabled()) {
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md_any, m2_md_any, b_md, r_md_any, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc.reset(new dnnl::matmul::desc(m1_md_any, m2_md_any, b_md, r_md_any));
    } else {
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md, m2_md, b_md, r_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc.reset(new dnnl::matmul::desc(m1_md, m2_md, b_md, r_md));
    }
  } else {
    if (dims == 2 && lazy_reorder_enabled()) {
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md_any, m2_md_any, r_md_any, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc.reset(new dnnl::matmul::desc(m1_md_any, m2_md_any, r_md_any));
    } else {
    #ifdef USE_PRIMITIVE_CACHE
      create_key(key, m1_md, m2_md, r_md, attr.beta_, attr.alpha_, post_flags);
    #endif
      matmul_desc.reset(new dnnl::matmul::desc(m1_md, m2_md, r_md));
    }
  }

#else // No USE_GEN12HP_ONEDNN
#ifdef USE_PRIMITIVE_CACHE
  create_key(key, m1_md, m2_md, r_md, attr.beta_, attr.alpha_, post_flags);
#endif
  matmul_desc.reset(new dnnl::matmul::desc(m1_md, m2_md, r_md));
#endif

  auto matmul_pd = dnnl::matmul::primitive_desc(*matmul_desc, pattr, engine);
#ifdef USE_PRIMITIVE_CACHE
  auto matmul_p = fetch_or_create_m<dnnl::matmul>(key, matmul_pd);
#else
  auto matmul_p = dnnl::matmul(matmul_pd);
#endif

#ifdef USE_GEN12HP_ONEDNN
  memory m1_usr_memory, m2_usr_memory, r_usr_memory;
  memory m1_memory, m2_memory, r_memory;
  Tensor r_;

  if (!lazy_reorder_enabled() || dims == 3) {
    m1_usr_memory = dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr());
    m2_usr_memory = dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr());
    r_usr_memory = dpcpp_onednn_memory(r_usr_md, engine, result.data_ptr());

    auto expected_m1_md = matmul_pd.src_desc();
    Tensor m1_;
    m1_memory = m1_usr_memory;
    if (m1_usr_memory.get_desc() != expected_m1_md) {
      m1_ = at::AtenIpexTypeXPU::empty({expected_m1_md.get_size() / m1.itemsize()},
      m1.options(), c10::nullopt);
      m1_memory = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, m1_usr_md, expected_m1_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m1_usr_memory, m1_memory);
#else
      auto reorder_p = dnnl::reorder(m1_usr_memory, m1_memory);
#endif
      DPCPP_ONEDNN_EXEC(reorder_p,
          strm, {{DNNL_ARG_FROM, m1_usr_memory}, {DNNL_ARG_TO, m1_memory}});
    }

    auto expected_m2_md = matmul_pd.weights_desc();
    Tensor m2_;
    m2_memory = m2_usr_memory;
    if (m2_usr_memory.get_desc() != expected_m2_md) {
      Tensor m2_opt;
      if (weight_cache_enabled()) {
        m2_opt = empty_opaque_tensor(expected_m2_md, m2.options(), c10::nullopt);
        m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_opt.data_ptr());
      } else {
        m2_ = at::AtenIpexTypeXPU::empty(
          {expected_m2_md.get_size() / m2.itemsize()}, m2.options(), c10::nullopt);
        m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_.data_ptr());
      }
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, m2_usr_md, expected_m2_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m2_usr_memory, m2_memory);
#else
      auto reorder_p = dnnl::reorder(m2_usr_memory, m2_memory);
#endif
      DPCPP_ONEDNN_EXEC(reorder_p,
          strm, {{DNNL_ARG_FROM, m2_usr_memory}, {DNNL_ARG_TO, m2_memory}});
    }

    auto expected_r_md = matmul_pd.dst_desc();
    r_memory = r_usr_memory;
    if (r_usr_memory.get_desc() != expected_r_md) {
      r_ = empty_opaque_tensor(expected_r_md, result.options(), c10::nullopt);
      r_memory = dpcpp_onednn_memory(expected_r_md, engine, r_.data_ptr());
      if (attr.beta_ != 1.f) {
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, r_usr_md, expected_r_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, r_usr_memory, r_memory);
#else
      auto reorder_p = dnnl::reorder(r_usr_memory, r_memory);
#endif
        DPCPP_ONEDNN_EXEC(reorder(r_usr_memory, r_memory),
            strm, {{DNNL_ARG_FROM, r_usr_memory}, {DNNL_ARG_TO, r_memory}});
      }
    }

  } else { // lazy_reorder_enabled || dims != 3

    auto m1_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m1);
    m1_usr_memory = m1_ctx.is_plain() ?
        dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr()) :
        dpcpp_onednn_memory({m1_ctx.meta()}, engine, m1.data_ptr());

    auto m2_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(m2);
    m2_usr_memory = m2_ctx.is_plain() ?
        dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr()) :
        dpcpp_onednn_memory({m2_ctx.meta()}, engine, m2.data_ptr());

    auto r_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(result);
    r_usr_memory = r_ctx.is_plain() ?
        dpcpp_onednn_memory(r_usr_md, engine, result.data_ptr()) :
        dpcpp_onednn_memory({r_ctx.meta()}, engine, result.data_ptr());

    auto expected_m1_md = matmul_pd.src_desc();
    Tensor m1_;
    m1_memory = m1_usr_memory;
    if (m1_usr_memory.get_desc() != expected_m1_md) {
      m1_ = at::AtenIpexTypeXPU::empty({expected_m1_md.get_size() / m1.itemsize()},
      m1.options(), c10::nullopt);
      m1_memory = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, m1_usr_md, expected_m1_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m1_usr_memory, m1_memory);
#else
      auto reorder_p = dnnl::reorder(m1_usr_memory, m1_memory);
#endif
      DPCPP_ONEDNN_EXEC(reorder_p,
          strm, {{DNNL_ARG_FROM, m1_usr_memory}, {DNNL_ARG_TO, m1_memory}});
    }

    auto expected_m2_md = matmul_pd.weights_desc();
    Tensor m2_;
    m2_memory = m2_usr_memory;
    if (m2_usr_memory.get_desc() != expected_m2_md) {
      Tensor m2_opt;
      if (weight_cache_enabled()) {
        m2_opt = empty_opaque_tensor(expected_m2_md, m2.options(), c10::nullopt);
        m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_opt.data_ptr());
      } else {
        m2_ = at::AtenIpexTypeXPU::empty(
          {expected_m2_md.get_size() / m2.itemsize()}, m2.options(), c10::nullopt);
        m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_.data_ptr());
      }
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, m2_usr_md, expected_m2_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m2_usr_memory, m2_memory);
#else
      auto reorder_p = dnnl::reorder(m2_usr_memory, m2_memory);
#endif
      DPCPP_ONEDNN_EXEC(reorder_p,
          strm, {{DNNL_ARG_FROM, m2_usr_memory}, {DNNL_ARG_TO, m2_memory}});

      if (weight_cache_enabled()) {
        strm.wait();
        // FIXME: thread safty
        auto m2_opt_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::
            release_tensor_ctx(m2_opt);
        at::AtenIpexTypeXPU::DPCPPTensorContext::
            set_tensor_ctx(m2, std::move(m2_opt_ctx));
      }
    }

    auto expected_r_md = matmul_pd.dst_desc();
    r_memory = r_usr_memory;
    if (r_usr_memory.get_desc() != expected_r_md) {
      r_ = empty_opaque_tensor(expected_r_md, result.options(), c10::nullopt);
      r_memory = dpcpp_onednn_memory(expected_r_md, engine, r_.data_ptr());
      if (attr.beta_ != 1.f) {
#ifdef USE_PRIMITIVE_CACHE
      create_key(key_r, r_usr_md, expected_r_md);
      auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, r_usr_memory, r_memory);
#else
      auto reorder_p = dnnl::reorder(r_usr_memory, r_memory);
#endif
        DPCPP_ONEDNN_EXEC(reorder(r_usr_memory, r_memory),
            strm, {{DNNL_ARG_FROM, r_usr_memory}, {DNNL_ARG_TO, r_memory}});
      }
    }
  }

  if (attr.beta_ == 1.f && attr.alpha_ == 1.f &&
      (!m1.is_quantized()) && (!m2.is_quantized())) {
    auto b_memory = dpcpp_onednn_memory(b_md, engine, b.data_ptr());
    DPCPP_ONEDNN_EXEC(matmul_p, strm,
      {{DNNL_ARG_SRC, m1_memory}, {DNNL_ARG_WEIGHTS, m2_memory},
        {DNNL_ARG_BIAS, b_memory}, {DNNL_ARG_DST, r_memory}});
  } else {
    DPCPP_ONEDNN_EXEC(matmul_p, strm,
      {{DNNL_ARG_SRC, m1_memory}, {DNNL_ARG_WEIGHTS, m2_memory},
        {DNNL_ARG_DST, r_memory}});
  }

#else // No USE_GEN12HP_ONEDNN

  memory m1_usr_memory, m2_usr_memory, r_usr_memory;
  memory m1_memory, m2_memory, r_memory;

  m1_usr_memory = dpcpp_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  m2_usr_memory = dpcpp_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  r_memory = dpcpp_onednn_memory(r_usr_md, engine, result.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  Tensor m1_;
  m1_memory = m1_usr_memory;
  if (m1_usr_memory.get_desc() != expected_m1_md) {
    m1_ = at::AtenIpexTypeXPU::empty({expected_m1_md.get_size() / m1.itemsize()},
    m1.options(), c10::nullopt);
    m1_memory = dpcpp_onednn_memory(expected_m1_md, engine, m1_.data_ptr());
#ifdef USE_PRIMITIVE_CACHE
    create_key(key_r, m1_usr_md, expected_m1_md);
    auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m1_usr_memory, m1_memory);
#else
    auto reorder_p = dnnl::reorder(m1_usr_memory, m1_memory);
#endif
    DPCPP_ONEDNN_EXEC(reorder_p,
        strm, {{DNNL_ARG_FROM, m1_usr_memory}, {DNNL_ARG_TO, m1_memory}});
  }

  auto expected_m2_md = matmul_pd.weights_desc();
  Tensor m2_;
  m2_memory = m2_usr_memory;
  if (m2_usr_memory.get_desc() != expected_m2_md) {
    Tensor m2_opt;
    if (weight_cache_enabled()) {
      m2_opt = empty_opaque_tensor(expected_m2_md, m2.options(), c10::nullopt);
      m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_opt.data_ptr());
    } else {
      m2_ = at::AtenIpexTypeXPU::empty(
        {expected_m2_md.get_size() / m2.itemsize()}, m2.options(), c10::nullopt);
      m2_memory = dpcpp_onednn_memory(expected_m2_md, engine, m2_.data_ptr());
    }
#ifdef USE_PRIMITIVE_CACHE
    create_key(key_r, m2_usr_md, expected_m2_md);
    auto reorder_p = fetch_or_create_m<dnnl::reorder>(key_r, m2_usr_memory, m2_memory);
#else
    auto reorder_p = dnnl::reorder(m2_usr_memory, m2_memory);
#endif
    DPCPP_ONEDNN_EXEC(reorder_p,
        strm, {{DNNL_ARG_FROM, m2_usr_memory}, {DNNL_ARG_TO, m2_memory}});
  }

  DPCPP_ONEDNN_EXEC(matmul_p, strm,
    {{DNNL_ARG_SRC, m1_memory}, {DNNL_ARG_WEIGHTS, m2_memory},
      {DNNL_ARG_DST, r_memory}});
#endif

#ifdef USE_GEN12HP_ONEDNN
  if (lazy_reorder_enabled() && r_memory != r_usr_memory && dims == 2) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(r_);
    DPCPPTensorContext::set_tensor_ctx(result, std::move(blk_ctx));
  }
#endif
}

bool check_broadcast(const Tensor& src, const IntArrayRef& shape){
  auto src_dim = src.dim();
  auto tgt_dim = shape.size();
  if (src_dim == 0 || src_dim > tgt_dim)
    return false;
  do {
    src_dim--;
    tgt_dim--;
    auto size = src.size(src_dim);
    if (size != 1 && size != shape[tgt_dim])
      return false;
  } while(src_dim);
  return true;
}

void gemm_broadcast(Tensor& result,
                    const Tensor& m1,
                    const Tensor& m2,
                    matmul_attr_t attr,
                    const Tensor bias = at::Tensor()) {
  std::vector<int64_t> result_shape;
  auto dim = m1.dim();
  if(m1.is_quantized()){
    if(m2.sizes()[1] == m1.sizes()[1]){
      m2.transpose_(0,1);
    }
  }
  if (dim == 2) {
    result_shape = attr.m2_trans_ ? std::vector<int64_t>{m1.size(0), m2.size(1)} :
    std::vector<int64_t>{m1.size(0), m2.size(0)};
  } else {
    result_shape = attr.m2_trans_ ? std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(2)} :
    std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(1)};
  }

  Tensor bc_bias = bias;
#ifdef USE_GEN12HP_ONEDNN
  if (bias.defined() && attr.beta_ && (attr.beta_ != 1.f || m1.is_quantized())) {
#else
  if (bias.defined() && attr.beta_) {
#endif
    TORCH_CHECK(check_broadcast(bias, result_shape),
                "bias ", bias.sizes(), " cannot broadcast to ", result_shape);
    std::tie(bc_bias) = expand_size(bias, result_shape, "gemm_broadcast");
    if (!result.is_same(bc_bias))
      result.resize_(bc_bias.sizes()).copy_(bc_bias);
  } else {
    result.resize_(result_shape);
  }

  dnnlGemmImpl(result, m1, m2, bc_bias, attr);
}
} // namespace impl


namespace AtenIpexTypeXPU {

using namespace impl;

Tensor& addmm_out(
        Tensor &result,
        const Tensor& self,
        const Tensor& m1,
        const Tensor& m2,
        Scalar beta,
        Scalar alpha) {
  matmul_attr_t attr(
          alpha.to<float>(),
          beta.to<float>(),
          0,
          true);
  checkBackend("addmm_out", {result, self, m1, m2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(self.size(0) ==  m1.size(0) && self.size(1) == m2.size(1),
              "size mismatch input ", self.sizes(), " m1 ", m1.sizes(), " m2 ", m2.sizes());

    impl::gemm_broadcast(
            result,
            m1,
            m2.scalar_type() == m1.scalar_type() ? m2 : m2.to(m1.scalar_type()),
            // bias convert to fp32 for accuracy when self is fp16 or bf16
            attr,
            // bias convert to fp32 for accuracy when self is fp16 or bf16
            self.scalar_type() == ScalarType::Half ||
            self.scalar_type() == ScalarType::BFloat16
            ? self.to(ScalarType::Float) : self);

  return result;
}

Tensor& addmm_(
    Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  matmul_attr_t attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("addmm_", {self, m1, m2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(self.size(0) ==  m1.size(0) && self.size(1) == m2.size(1),
              "size mismatch input ", self.sizes(), " m1 ", m1.sizes(), " m2 ", m2.sizes());

  impl::gemm_broadcast(
  self,
  m1,
  m2.scalar_type() == m1.scalar_type() ? m2 :
                     (m1.scalar_type() == ScalarType::BFloat16 || m2.scalar_type() == ScalarType::BFloat16) ? m2 :
                      m2.to(m1.scalar_type()),
  // bias convert to fp32 for accuracy when self is fp16 or bf16
  attr,
  self.scalar_type() == ScalarType::Half ||
          self.scalar_type() == ScalarType::BFloat16
      ? self.to(ScalarType::Float) : self);

  return self;
}

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  matmul_attr_t attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);

  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  checkBackend("addmm", {input, m1, m2}, Backend::XPU);

  Tensor result;
  if (m1.scalar_type() == at::ScalarType::BFloat16){
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(
  result,
  m1,
  m2.scalar_type() == m1.scalar_type() ? m2 :
                     (m1.scalar_type() == ScalarType::BFloat16 || m2.scalar_type() == ScalarType::BFloat16) ? m2 :
                      m2.to(m1.scalar_type()),
  // bias convert to fp32 for accuracy when input is fp16 or bf16
  attr,
  input.scalar_type() == ScalarType::Half ||
          input.scalar_type() == ScalarType::BFloat16
      ? input.to(ScalarType::Float) : input);

  return result;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  matmul_attr_t attr(1.f, 0.f, 0, true);
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "expected 2D tensor");

  auto self_dt = self.scalar_type();
  auto result_dt = result.scalar_type();
  auto mat2_dt = mat2.scalar_type();

  impl::gemm_broadcast(
  result,
  (self_dt == result_dt ||
   ((self_dt == ScalarType::BFloat16 && result_dt != ScalarType::BFloat16) || (result_dt == ScalarType::BFloat16 && self_dt != ScalarType::BFloat16))) ? self : self.to(result_dt),
  (mat2_dt == result_dt || 
   ((mat2_dt == ScalarType::BFloat16 && result_dt != ScalarType::BFloat16) || (result_dt == ScalarType::BFloat16 && mat2_dt != ScalarType::BFloat16))) ? mat2 : mat2.to(result_dt),
  attr);

  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::mm_out(result, self, mat2);
  return result;
}

Tensor& baddbmm_(
  Tensor& self,
  const Tensor& batch1,
  const Tensor& batch2,
  Scalar beta,
  Scalar alpha) {
  matmul_attr_t attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("baddbmm_", {self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(self.size(0) == batch1.size(0) && \
              self.size(1) == batch1.size(1) && \
              self.size(2) == batch2.size(2),
              "size mismatch input ", self.sizes(),
              " batch1 ", batch1.sizes(), " batch2 ", batch2.sizes());
  impl::gemm_broadcast(self, batch1, batch2, attr, self);

  return self;
}

Tensor& baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  matmul_attr_t attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      true);
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, batch1, batch2, attr, input);

  return result;
}

Tensor baddbmm(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor r = at::empty({0}, input.options());
  at::AtenIpexTypeXPU::baddbmm_out(r, input, batch1, batch2, beta, alpha);
  return r;
}

Tensor& addbmm_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {

  checkBackend("addbmm_out", {out, self, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  Tensor b1;
  if (batch1.size(0) > 1) {
    b1 = batch1.transpose(0, 1).contiguous().view({batch1.size(1), -1});
  } else {
    b1 = batch1.view({batch1.size(1), -1});
  }
  auto b2 = batch2.view({-1, batch2.size(2)});
  at::AtenIpexTypeXPU::addmm_out(out, self, b1, b2, beta, alpha);

  return out;
}

Tensor& addbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  at::AtenIpexTypeXPU::addbmm_out(self, self, batch1, batch2, beta, alpha);
  return self;
}

Tensor addbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::addbmm_out(out, self, batch1, batch2, beta, alpha);
  return out;
}

Tensor& bmm_out(Tensor& result, const Tensor& self, const Tensor& batch2) {
  matmul_attr_t attr(1, 0, 0, true);
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, self, batch2, attr);

  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::bmm_out(result, self, batch2);
  return result;
}

// FIXME: should not be here
Tensor linear_relu(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  matmul_attr_t attr(
      1.f,
      1.f,
      matmul_attr_t::kind_with_relu,
      false);
  RECORD_FUNCTION("linear_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_relu", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());

    impl::gemm_broadcast(result, input, weight, attr, bias);

    return result;
  }

  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::relu(output);
}

Tensor linear_sigmoid(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  matmul_attr_t attr(
      1.f,
      1.f,
      matmul_attr_t::kind_with_sigmoid,
      false);
  RECORD_FUNCTION("linear_sigmoid",
                  std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_sigmoid", {input, weight, bias}, Backend::XPU);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());
    impl::gemm_broadcast(result, input, weight, attr, bias);

    return result;
  }
  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::sigmoid(output);

}

Tensor trans_linear(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2) {
  matmul_attr_t attr(
      1.f,
      1.f,
      0,
      false);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");

  if(m1.is_quantized()){
    checkBackend("addmm", m1, Backend::QuantizedXPU);
  } else {
    checkBackend("addmm", {input, m1, m2}, Backend::XPU);
    TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  }

  Tensor result;
  if(input.is_quantized()){
    result = _empty_affine_quantized({0},
              device(kXPU).dtype(input.scalar_type()),
              1.f,
              static_cast<int>(0),
              MemoryFormat::Contiguous);
  } else if (m1.scalar_type() == at::ScalarType::BFloat16){
    // align with bf16 input
    result = at::empty({0}, m1.options());
  } else {
    result = at::empty({0}, input.options());
  }

  if(m1.is_quantized()){
    impl::gemm_broadcast(result, m1, m2, attr, input);
  } else {
    impl::gemm_broadcast(
    result,
    m1,
    m2.scalar_type() == m1.scalar_type() ? m2 : m2.to(m1.scalar_type()),
    // bias convert to fp32 for accuracy when input is fp16 or bf16
    attr,
    input.scalar_type() == ScalarType::Half ||
            input.scalar_type() == ScalarType::BFloat16
        ? input.to(ScalarType::Float) : input);

  }

  return result;
}

Tensor addmv(
    const Tensor & self,
    const Tensor & mat,
    const Tensor & vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) ==  vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  Tensor result = at::AtenIpexTypeXPU::addmm(self_v, mat, vec_v, beta, alpha);
  return result.view({mat.size(0)});
}

Tensor& addmv_(
    Tensor & self,
    const Tensor & mat,
    const Tensor & vec,
    at::Scalar beta,
    at::Scalar alpha) {
  TORCH_CHECK(self.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(vec.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(mat.size(1) ==  vec.size(0));

  Tensor vec_v = vec.view({vec.size(0), 1});
  Tensor self_v = self.view({self.size(0), 1});
  at::AtenIpexTypeXPU::addmm_(self_v, mat, vec_v, beta, alpha);
  return self;
}

Tensor matmul_sum(
    Tensor& accumu,
    const Tensor& m1,
    const Tensor& m2,
    at::Scalar beta) {
  Tensor result, bias;

  TORCH_CHECK(m1.dim() == 2 || m2.dim() == 2, "expected 2D tensor");
  if (accumu.dim() == 1) {
    if (beta.to<float>() == 1.0f) {
      result = at::empty({0}, m1.options());
      bias = accumu;
    } else {
      std::tie(result) = expand_size(
          accumu, m1.dim() == 2 ? m1.sizes() : m2.sizes());
    }
  } else {
    result = accumu;
  }

  // collaps a,b,c to axb,c for m1
  // FIXME: no m2 to collaps so far
  std::vector<int64_t> m1_shape, r_shape;
  if (m1.dim() != 2) {
    for (int i = 0; i < m1.sizes().size() - 1; i++) {
      m1_shape.push_back(m1.sizes()[i]);
      r_shape.push_back(m1.sizes()[i]);
    }
    m1_shape.push_back(m1.sizes()[m1.sizes().size() - 1]);
    r_shape.push_back(m2.sizes()[1]);

    std::vector<int64_t> sizes = m1.sizes().vec();
    std::vector<int64_t> strides = m1.strides().vec();
    at::collapse_dims(sizes.data(), strides.data(), m1.dim(), m1.dim() - 1);
    m1.resize_({sizes.data()[0], sizes.data()[1]});
  }

  matmul_attr_t attr(
      1.f,
      beta.to<float>(),
      0,
      true);

  impl::gemm_broadcast(
      result,
      m1,
      m2.scalar_type() == m1.scalar_type() ? m2 : m2.to(m1.scalar_type()),
      attr,
      bias);

  if (r_shape.size()) {
    m1.resize_(m1_shape);
    result.resize_(r_shape);
  }

  return result;
}

Tensor& trans_baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  matmul_attr_t attr(
      alpha.to<float>(),
      beta.to<float>(),
      0,
      false);
  checkBackend("trans_baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  impl::gemm_broadcast(result, batch1, batch2, attr, input);

  return result;
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor addmm(
  const Tensor& input,
  const Tensor& m1,
  const Tensor& m2,
  Scalar beta,
  Scalar alpha) {
  matmul_attr_t attr(
          alpha.to<float>(),
          beta.to<float>(),
          0,
          true);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");

  checkBackend("addmm", m1, Backend::QuantizedXPU);

  Tensor result;
  if(input.is_quantized()){
    result = _empty_affine_quantized({0},
                                     device(kXPU).dtype(input.scalar_type()),
                                     1.f,
                                     static_cast<int>(0),
                                     MemoryFormat::Contiguous);
  } else {
    result = at::empty({0}, input.options());
  }

  impl::gemm_broadcast(result, m1, m2, attr, input);

  return result;
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
