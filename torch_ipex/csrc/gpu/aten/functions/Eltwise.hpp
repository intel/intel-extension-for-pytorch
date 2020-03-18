#pragma once
#include <ATen/ATen.h>
#include <ATen/Config.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

using namespace mkldnn;
using mkldnn::algorithm;

using namespace at::dpcpp;

namespace at { namespace dpcpp {

template <algorithm alg_kind>
void dpcpp_eltwise (at::Tensor& output, const at::Tensor& input, float alpha, float beta)
{
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {n, ic, ih, iw};
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);

  eltwise_forward::desc eltwise_eltwiseFwd_desc(prop_kind::forward, alg_kind, input_md, alpha, beta);
  auto eltwise_forward_pd = eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto output_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);


  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<mkldnn::primitive> eltwise_fwd;
  eltwise_fwd.reset(new mkldnn::eltwise_forward(eltwise_forward_pd));
  eltwise_fwd->execute(strm, {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_DST, output_usr_memory}});
}

template <algorithm alg_kind>
void dpcpp_eltwise_backward(char* diff_src, char* src, char* diff_dst, int32_t len, float alpha, float beta)
{
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

  eltwise_forward::desc eltwise_eltwiseFwd_desc(prop_kind::forward_training, alg_kind, src_md, alpha, beta);
  auto eltwise_forward_pd = eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);
  eltwise_backward::desc eltwise_reluBwd_desc(alg_kind, diff_dst_md, src_md, alpha, beta);
  auto eltwise_backward_pd = eltwise_backward::primitive_desc(eltwise_reluBwd_desc, engine, eltwise_forward_pd);

  auto src_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(src, src_usr_memory);

  auto diff_dst_memory  = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(diff_dst, diff_dst_memory);

  auto diff_src_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(diff_src, diff_src_memory);

  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<mkldnn::primitive> eltwise_bwd;
  eltwise_bwd.reset(new mkldnn::eltwise_backward(eltwise_backward_pd));
  eltwise_bwd->execute(strm, {
      {MKLDNN_ARG_SRC, src_usr_memory},
      {MKLDNN_ARG_DIFF_DST, diff_dst_memory},
      {MKLDNN_ARG_DIFF_SRC, diff_src_memory}});
}

}} // at::dpcpp


