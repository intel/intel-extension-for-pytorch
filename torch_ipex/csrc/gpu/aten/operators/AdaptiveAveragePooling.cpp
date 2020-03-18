#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>

using namespace mkldnn;
using namespace at::dpcpp;
namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static void adaptive_avg_pool2d_out_dpcpp_frame(
    scalar_t *input_data, scalar_t *output_data, int64_t nbatch,
    int64_t nInputPlane, int64_t inputWidth, int64_t inputHeight,
    int64_t outputWidth, int64_t outputHeight, int kW, int kH, int dW, int dH,
    int padW, int padH, algorithm alg_kind, prop_kind prop_kind) {
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = memory::data_type::f32;
  if (std::is_same<scalar_t, Half>::value == true) {
    data_t = memory::data_type::f16;
    prop_kind = dnnl::prop_kind::forward_inference;
  }
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
  memory::dims output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
  memory::dims kernel = {kH, kW};
  memory::dims stride = {dH, dW};
  memory::dims padding = {padH, padW};

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto output_md = memory::desc({output_tz}, data_t, format_nchw);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(input_data, input_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(output_data, output_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(
      new pooling_forward::desc(prop_kind, alg_kind, input_md, output_md,
                                stride, kernel, padding, padding));

  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  // auto input_d = pooling_forward_pd->src_desc();
  auto input_memory = input_usr_memory;

  // Currently, DPCPP path doesn't support internal format.
  // input has the same format with input_usr.
  // if (input_usr_memory.get_desc() != input_d) {
  //   input_memory = memory(input_d, engine);
  //   reorder(input_usr_memory, input_memory).
  //       execute(strm, input_usr_memory, input_memory);
  // }

  // auto output_d = pooling_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;

  // output has the same format with output_usr.
  // if (output_usr_memory.get_desc() != output_d) {
  //   output_memory = memory(output_d, engine);
  // }

  std::shared_ptr<pooling_forward> pool_forward;
  pool_forward.reset(new pooling_forward(*pooling_forward_pd));
  pool_forward->execute(
      strm, {{MKLDNN_ARG_SRC, input_memory}, {MKLDNN_ARG_DST, output_memory}});

  // reorder output
  // if (output_memory != output_usr_memory) {
  //   reorder(output_memory, output_usr_memory).
  //       execute(strm, output_memory, output_usr_memory);
  // }
}

template <typename scalar_t>
static void adaptive_avg_pool2d_backward_out_dpcpp_frame(
    scalar_t *gradInput_data, scalar_t *gradOutput_data, int64_t nbatch,
    int64_t nInputPlane, int64_t inputWidth, int64_t inputHeight,
    int64_t outputWidth, int64_t outputHeight, int kW, int kH, int dW, int dH,
    int padW, int padH, algorithm alg_kind, prop_kind prop_kind) {
  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
  memory::dims output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
  memory::dims kernel = {kH, kW};
  memory::dims stride = {dH, dW};
  memory::dims padding = {padH, padW};

  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto output_md = memory::desc({output_tz}, data_t, format_nchw);

  auto diff_dst_usr_memory =
      memory({{{output_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(gradOutput_data, diff_dst_usr_memory);

  auto diff_src_usr_memory =
      memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(gradInput_data, diff_src_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(
      new pooling_forward::desc(prop_kind, alg_kind, input_md, output_md,
                                stride, kernel, padding, padding));
  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  std::shared_ptr<pooling_backward::desc> pooling_backward_desc;
  pooling_backward_desc.reset(new pooling_backward::desc(
      alg_kind, input_md, output_md, stride, kernel, padding, padding));
  std::shared_ptr<pooling_backward::primitive_desc> pooling_backward_pd;
  pooling_backward_pd.reset(new pooling_backward::primitive_desc(
      *pooling_backward_desc, engine, *pooling_forward_pd));

  // auto diff_dst_md = pooling_backward_pd->diff_dst_desc();
  auto diff_dst_memory = diff_dst_usr_memory;

  // Currently, DPCPP path doesn't support internal format.
  // diff_dst has the same format with dst.
  // if (diff_dst_usr_memory.get_desc() != diff_dst_md) {
  //   diff_dst_memory = memory(diff_dst_md, engine);
  //   reorder(diff_dst_usr_memory, diff_dst_memory).
  //       execute(strm, diff_dst_usr_memory, diff_dst_memory);
  // }

  // auto diff_src_md = pooling_backward_pd->diff_src_desc();
  auto diff_src_memory = diff_src_usr_memory;

  // diff_src has the same format with src.
  // if (diff_src_usr_memory.get_desc() != diff_src_md) {
  //   diff_src_memory = memory(diff_src_md, engine);
  // }

  std::shared_ptr<pooling_backward> pool_backward;
  pool_backward.reset(new pooling_backward(*pooling_backward_pd));

  pool_backward->execute(strm, {{MKLDNN_ARG_DIFF_DST, diff_dst_memory},
                                {MKLDNN_ARG_DIFF_SRC, diff_src_memory}});

  // Reorder diff_src
  // if (diff_src_memory != diff_src_usr_memory) {
  //   reorder(diff_src_memory, diff_src_usr_memory).
  //       execute(strm, diff_src_memory, diff_src_usr_memory);
  // }
}

void adaptive_avg_pool2d_out_dpcpp_template(Tensor &output, const Tensor &input,
                                            IntArrayRef output_size) {
  TORCH_CHECK((input.ndimension() == 4),
              "only support 4 dims on DPCPP device now!");
  int kW, kH, dW, dH;
  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int padW = 0;
  int padH = 0;
  // bool ceil_mode = false;
  int64_t nOutputCols = output_size[1];
  int64_t nOutputRows = output_size[0];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

  TORCH_CHECK((nInputRows % nOutputRows == 0), "row input size is not "
                                               "divisible by the output size "
                                               "is not supported yet");
  TORCH_CHECK((nInputCols % nOutputCols == 0), "column input size is not "
                                               "divisible by the output size "
                                               "is not supported yet");

  kW = nInputCols / nOutputCols;
  kH = nInputRows / nOutputRows;
  dW = kW;
  dH = kH;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = dnnl::prop_kind::forward_training;

  output.resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool2d_dpcpp", [&] {
        auto input_data = input.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        adaptive_avg_pool2d_out_dpcpp_frame<scalar_t>(
            input_data, output_data, batchSize, nInputPlane, nInputCols,
            nInputRows, nOutputCols, nOutputRows, kW, kH, dW, dH, padW, padH,
            alg_kind, prop_kind);
      });
}

void adaptive_avg_pool2d_backward_out_dpcpp_template(Tensor &gradInput,
                                                     const Tensor &gradOutput,
                                                     const Tensor &input) {
  TORCH_CHECK((input.ndimension() == 4),
              "only support 4 dims on DPCPP device now!");
  int kW, kH, dW, dH;
  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int padW = 0;
  int padH = 0;
  auto output_size_vec = gradOutput.sizes();
  int64_t nOutputCols = output_size_vec[3];
  int64_t nOutputRows = output_size_vec[2];

  // Input is NCHW format
  nInputCols = input.size(3);
  nInputRows = input.size(2);
  nInputPlane = input.size(1);
  batchSize = input.size(0);

  TORCH_CHECK((nInputRows % nOutputRows == 0), "row input size is not "
                                               "divisible by the output size "
                                               "is not supported yet");
  TORCH_CHECK((nInputCols % nOutputCols == 0), "column input size is not "
                                               "divisible by the output size "
                                               "is not supported yet");

  kW = nInputCols / nOutputCols;
  kH = nInputRows / nOutputRows;
  dW = kW;
  dH = kH;

  auto alg_kind = algorithm::pooling_avg;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "adaptive_avg_pool2d_backward_dpcpp", [&] {
        auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
        auto gradInput_data = gradInput.data_ptr<scalar_t>();
        adaptive_avg_pool2d_backward_out_dpcpp_frame<scalar_t>(
            gradInput_data, gradOutput_data, batchSize, nInputPlane, nInputCols,
            nInputRows, nOutputCols, nOutputRows, kW, kH, dW, dH, padW, padH,
            alg_kind, prop_kind);
      });
}

Tensor &adaptive_avg_pool2d_out_dpcpp(Tensor &output, const Tensor &input,
                                      IntArrayRef output_size) {
  adaptive_avg_pool2d_out_dpcpp_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool2d_dpcpp(at::Tensor const &input,
                                 IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out_dpcpp_template(output, input, output_size);
  return output;
}

Tensor &adaptive_avg_pool2d_backward_out_dpcpp(Tensor &gradInput,
                                               const Tensor &gradOutput,
                                               const Tensor &input) {
  gradInput.resize_as_(input);
  adaptive_avg_pool2d_backward_out_dpcpp_template(gradInput, gradOutput, input);
  return gradInput;
}

Tensor adaptive_avg_pool2d_backward_dpcpp(const Tensor &gradOutput,
                                          const Tensor &input) {
  auto gradInput = at::zeros_like(input);
  adaptive_avg_pool2d_backward_out_dpcpp_template(gradInput, gradOutput, input);
  return gradInput;
}

} // namespace impl

Tensor &adaptive_avg_pool2d_out(Tensor &out, const Tensor &self,
                                IntArrayRef output_size) {
  impl::adaptive_avg_pool2d_out_dpcpp(out, self, output_size);
  return out;
}

Tensor _adaptive_avg_pool2d(const Tensor &self, IntArrayRef output_size) {
  return impl::adaptive_avg_pool2d_dpcpp(self, output_size);
}

Tensor adaptive_avg_pool2d(const Tensor &self, IntArrayRef output_size) {
  return impl::adaptive_avg_pool2d_dpcpp(self, output_size);
}

Tensor _adaptive_avg_pool2d_backward(const Tensor &grad_output,
                                     const Tensor &self) {
  return impl::adaptive_avg_pool2d_backward_dpcpp(grad_output, self);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
