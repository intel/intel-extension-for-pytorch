#pragma once

#include <core/Runtime.h>
#include <dnnl.hpp>

using namespace mkldnn;

namespace at {
namespace dpcpp {

void inner_product(
    int M,
    int N,
    int K,
    void* output,
    void* input,
    void* weight,
    Tensor bias,
    bool use_bias) {
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int32_t n = M;
  int32_t ic = K;
  int32_t oc = N;
  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_nc = memory::format_tag::nc;
  auto format_oi = memory::format_tag::oi;
  auto format_x = memory::format_tag::x;

  memory::dims input_tz = {n, ic};
  memory::dims weight_tz = {oc, ic};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  std::shared_ptr<inner_product_forward::desc> ipFwd_desc;
  if (use_bias) {
    ipFwd_desc.reset(new inner_product_forward::desc(
        prop_kind::forward_inference, input_md, weight_md, bias_md, output_md));
  } else {
    ipFwd_desc.reset(new inner_product_forward::desc(
        prop_kind::forward_inference, input_md, weight_md, output_md));
  }
  auto ip_forward_pd =
      inner_product_forward::primitive_desc(*ipFwd_desc, engine);

  auto input_usr_memory =
      dpcpp_onednn_memory({{input_tz}, data_t, format_nc}, engine, input);
  auto weight_usr_memory =
      dpcpp_onednn_memory({{weight_tz}, data_t, format_oi}, engine, weight);
  auto output_usr_memory =
      dpcpp_onednn_memory({{output_tz}, data_t, format_nc}, engine, output);

  auto strm = GpuStreamManager::Instance().get_stream();
  memory bias_usr_memory;
  if (use_bias) {
    bias_usr_memory = dpcpp_onednn_memory(
        {{bias_tz}, data_t, format_x}, engine, bias.data_ptr());
  } else {
    // dummy dnnl::memory
    bias_usr_memory = memory({{{}, data_t, format_x}, engine});
  }

  auto ip_forward = inner_product_forward(ip_forward_pd);
  DPCPP_ONEDNN_EXEC(
      ip_forward,
      strm,
      {{MKLDNN_ARG_SRC, input_usr_memory},
       {MKLDNN_ARG_WEIGHTS, weight_usr_memory},
       {MKLDNN_ARG_BIAS, bias_usr_memory},
       {MKLDNN_ARG_DST, output_usr_memory}});
}
} // namespace dpcpp
} // namespace at
