#pragma once

#include <core/Runtime.h>
#include <dnnl.hpp>


using namespace mkldnn;

namespace at { namespace dpcpp {

template <typename data_type, bool use_bias>
void mkldnn_inner_product(int M, int N, int K, data_type* output, data_type* input, data_type* weight, data_type* bias)
{
#ifndef DNNL_CPU_ONLY
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

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
    ipFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, input_md, weight_md, bias_md, output_md));
  }
  else {
    ipFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, input_md, weight_md, output_md));
  }
  auto ip_forward_pd = inner_product_forward::primitive_desc(*ipFwd_desc, engine);

// #if AT_SYCL_ENABLED()
#if 1
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nc}, engine});
  sycl_set_mkldnn_buffer(input, input_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_oi}, engine});
  sycl_set_mkldnn_buffer(weight, weight_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nc}, engine});
  sycl_set_mkldnn_buffer(output, output_usr_memory);

#else


#endif

  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<inner_product_forward> ip_forward;
  std::shared_ptr<memory> bias_usr_memory;
  if (use_bias) {
// #if AT_SYCL_ENABLED()
#if 1
    bias_usr_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine}));
    sycl_set_mkldnn_buffer(bias, *bias_usr_memory);
#else
#endif
  } else {
    bias_usr_memory.reset(new memory({{{}, data_t, format_x}, engine}));
  }

  ip_forward.reset(new inner_product_forward(ip_forward_pd));
  ip_forward->execute(strm, {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_WEIGHTS, weight_usr_memory},
      {MKLDNN_ARG_BIAS, *bias_usr_memory},
      {MKLDNN_ARG_DST, output_usr_memory}});
}
}} // namespace at::native

