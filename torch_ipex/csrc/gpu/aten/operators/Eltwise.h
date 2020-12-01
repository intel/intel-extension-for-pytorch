#pragma once
#include <ATen/ATen.h>
#include <ATen/Config.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include <ATen/ipex_type_dpcpp_customized.h>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

using namespace dnnl;
using dnnl::algorithm;
using namespace at::dpcpp;

namespace at {
namespace dpcpp {

template <algorithm alg_kind>
void dpcpp_eltwise(
    at::Tensor& output,
    const at::Tensor& input,
    float alpha,
    float beta) {
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  std::vector<int64_t> dims;
  for (size_t i = 0; i < input.dim(); i++) {
    dims.push_back(input.size(i));
  }

  memory::dims input_tz = dims;
  auto data_t = dt_to_dnnl(input.scalar_type());
  auto format_any = get_dnnl_default_format(input.dim());
  auto input_md = memory::desc({input_tz}, data_t, format_any);

  memory input_usr_memory;
  if (!lazy_reorder_enabled()) {
    input_usr_memory = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  } else {
    auto input_ctx = at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(input);
    input_md = input_ctx.is_plain() ? input_md : input_ctx.meta();
    input_usr_memory = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  }


#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, input_md, alpha, beta);
#endif

  eltwise_forward::desc eltwise_eltwiseFwd_desc(prop_kind::forward, alg_kind, input_md, alpha, beta);
  auto eltwise_forward_pd = eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);

  memory output_usr_memory;
  if (!lazy_reorder_enabled()) {
    if (!output.defined())
      output = at::empty_like(input);
    output_usr_memory = dpcpp_onednn_memory(eltwise_forward_pd.dst_desc(), engine, output.data_ptr());
  } else {
    if (output.defined()) {
      auto output_ctx = at::AtenIpexTypeDPCPP::DPCPPTensorContext::get_tensor_ctx(output);
      auto output_md = output_ctx.is_plain() ? input_md : output_ctx.meta();
      output_usr_memory = dpcpp_onednn_memory(output_md, engine, output.data_ptr());
    } else {
      auto plain_output_md = memory::desc({input_tz}, data_t, format_any);
      auto expected_output_md = eltwise_forward_pd.dst_desc();
      if (plain_output_md != expected_output_md) {
        output = at::AtenIpexTypeDPCPP::empty_opaque_tensor(
            expected_output_md, input.options(), c10::nullopt);
        output_usr_memory = dpcpp_onednn_memory(expected_output_md, engine, output.data_ptr());
      } else {
        output = at::empty_like(input);
        output_usr_memory = dpcpp_onednn_memory(plain_output_md, engine, output.data_ptr());
      }
    }
  }

  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  auto eltwise_fwd =
      fetch_or_create_m<dnnl::eltwise_forward>(key, eltwise_forward_pd);
#else
  auto eltwise_fwd = dnnl::eltwise_forward(eltwise_forward_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      eltwise_fwd,
      strm,
      {{DNNL_ARG_SRC, input_usr_memory},
       {DNNL_ARG_DST, output_usr_memory}});
}

template <algorithm alg_kind>
void dpcpp_eltwise_backward(
    char* diff_src,
    char* src,
    char* diff_dst,
    int32_t len,
    float alpha,
    float beta) {
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int32_t n = len;
  int32_t ic = 1;
  int32_t ih = 1;
  int32_t iw = 1;
  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {n, ic, ih, iw};
  auto src_md = memory::desc({input_tz}, data_t, format_nchw);
  auto diff_dst_md = memory::desc({input_tz}, data_t, format_nchw);

  eltwise_forward::desc eltwise_eltwiseFwd_desc(
      prop_kind::forward_training, alg_kind, src_md, alpha, beta);
  auto eltwise_forward_pd =eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);
  eltwise_backward::desc eltwise_reluBwd_desc(alg_kind, diff_dst_md, src_md, alpha, beta);
  auto eltwise_backward_pd = eltwise_backward::primitive_desc(
      eltwise_reluBwd_desc, engine, eltwise_forward_pd);

  auto src_usr_memory = dpcpp_onednn_memory({{input_tz}, data_t, format_nchw}, engine, src);
  auto diff_dst_memory = dpcpp_onednn_memory({{input_tz}, data_t, format_nchw}, engine, diff_dst);
  auto diff_src_memory = dpcpp_onednn_memory({{input_tz}, data_t, format_nchw}, engine, diff_src);

  auto strm = GpuStreamManager::Instance().get_stream();
  auto eltwise_bwd = dnnl::eltwise_backward(eltwise_backward_pd);
  DPCPP_ONEDNN_EXEC(eltwise_bwd, strm,
      {{DNNL_ARG_SRC, src_usr_memory},
       {DNNL_ARG_DIFF_DST, diff_dst_memory},
       {DNNL_ARG_DIFF_SRC, diff_src_memory}});
}
} // namespace dpcpp
} // namespace at
