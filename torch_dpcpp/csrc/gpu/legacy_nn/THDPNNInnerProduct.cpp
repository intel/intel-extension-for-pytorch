#include <ATen/dpcpp/Runtime.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <THDPNN/THDPNNInnerProduct.h>

void dnnl_vec_inner_product_forward(
    int K,
    THSYCLTensor *input1,
    THSYCLTensor *input2,
    THSYCLTensor *output) {
  at::Device curDevice = at::Device(at::kDPCPP, c10::sycl::current_device());
  auto engine = at::native::GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = at::native::GpuStreamManager::Instance().get_stream();

  int32_t ic = K;
  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_nc = memory::format_tag::nc;
  auto format_oi = memory::format_tag::oi;
  auto format_x = memory::format_tag::x;

  memory::dims input_tz = {1, ic};
  memory::dims weight_tz = {1, ic};
  memory::dims output_tz = {1, 1};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  std::shared_ptr<inner_product_forward::desc> ipFwd_desc;
  ipFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, input_md, weight_md, output_md));
  auto ip_forward_pd = inner_product_forward::primitive_desc(*ipFwd_desc, engine);

  // vec_inner_product only supports fp32
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nc}, engine});
  at::native::sycl_set_mkldnn_buffer(input1->data<float>(), input_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_oi}, engine});
  at::native::sycl_set_mkldnn_buffer(input2->data<float>(), weight_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nc}, engine});
  at::native::sycl_set_mkldnn_buffer(output->data<float>(), output_usr_memory);

  std::shared_ptr<inner_product_forward> ip_forward;
  std::shared_ptr<memory> bias_usr_memory;

  bias_usr_memory.reset(new memory({{{}, data_t, format_x}, engine}));

  ip_forward.reset(new inner_product_forward(ip_forward_pd));
  ip_forward->execute(strm, {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_WEIGHTS, weight_usr_memory},
      {MKLDNN_ARG_BIAS, *bias_usr_memory},
      {MKLDNN_ARG_DST, output_usr_memory}});

}

