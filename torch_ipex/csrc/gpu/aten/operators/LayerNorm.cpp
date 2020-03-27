#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <core/Memory.h>
#include <core/Runtime.h>
#include <utils/Math.h>

using namespace dnnl;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& X,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    double epsilon) {
  auto input = X.contiguous().view({1, M, N});
  auto output = at::empty_like(input);

  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // FP16 Data Type only support forward_inference
  bool training = input.scalar_type() == ScalarType::Half ? false : true;
  auto propagation = training ? dnnl::prop_kind::forward_training
                              : dnnl::prop_kind::forward_inference;

  int32_t n, ic, ih;
  n = input.size(0);
  ic = input.size(1);
  ih = input.size(2);

  auto data_t = dt_to_dnnl(input.scalar_type());
  auto format_nch = memory::format_tag::abc;

  memory::dims input_tz = {n, ic, ih};
  auto input_md = memory::desc({input_tz}, data_t, format_nch);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nch}, engine});
  dpcpp_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto output_usr_memory = memory({{{input_tz}, data_t, format_nch}, engine});
  dpcpp_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

  normalization_flags flags = normalization_flags::use_scale_shift;

  bool useScaleShift = (bool)(flags & normalization_flags::use_scale_shift);
  layer_normalization_forward::desc layer_norm_forward_desc(
      propagation, input_md, epsilon, flags);
  auto lnorm_fwd_pd = layer_normalization_forward::primitive_desc(
      layer_norm_forward_desc, engine);

  std::unordered_map<int, memory> args = {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_DST, output_usr_memory},
  };

  Tensor mean = at::empty({n * ih * ic}, input.options());
  Tensor rstd = at::empty({n * ih * ic}, input.options());
  if (training) {
    auto mean_memory = memory(lnorm_fwd_pd.mean_desc(), engine);
    auto var_memory = memory(lnorm_fwd_pd.variance_desc(), engine);
    dpcpp_set_mkldnn_buffer(mean.data_ptr(), mean_memory);
    dpcpp_set_mkldnn_buffer(rstd.data_ptr(), var_memory);
    args.insert({DNNL_ARG_MEAN, mean_memory});
    args.insert({DNNL_ARG_VARIANCE, var_memory});
  }

  if (useScaleShift) {
    auto weight_bias_memory = memory(lnorm_fwd_pd.weights_desc(), engine);
    auto weight_bias = at::empty(2 * ih, weight.options());
    dpcppMemcpyAsync(
        weight_bias.data_ptr(),
        weight.data_ptr(),
        ih * sizeof(float),
        DeviceToDevice);
    dpcppMemcpyAsync(
        static_cast<uint8_t*>(weight_bias.data_ptr()) + ih * sizeof(float),
        bias.data_ptr(),
        ih * sizeof(float),
        DeviceToDevice);
    dpcpp_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);
    args.insert({DNNL_ARG_SCALE_SHIFT, weight_bias_memory});
  }

  auto strm = GpuStreamManager::Instance().get_stream();
  layer_normalization_forward(lnorm_fwd_pd).execute(strm, args);

  return std::make_tuple(output.view(X.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  auto input = X.contiguous().view({1, M, N});
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  double epsilon = 1e-5;

  if (grad_input_mask[0]) {
    grad_input = at::empty_like(input);
  }
  if (grad_input_mask[1]) {
    grad_weight = at::empty_like(weight);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty_like(weight);
  }

  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto flags = normalization_flags::use_scale_shift;
  bool useScaleShift = (bool)(flags & normalization_flags::use_scale_shift);

  int32_t n;
  int32_t ic;
  int32_t ih;
  int32_t iw;

  n = input.size(0);
  ic = input.size(1);
  ih = input.size(2);
  iw = 1 /*input.size(3)*/;

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::abc;

  memory::dims input_tz = {n, ic, ih};
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto grad_output_md = input_md;

  layer_normalization_forward::desc layer_norm_forward_desc(
      dnnl::prop_kind::forward_training, input_md, epsilon, flags);
  auto lnorm_fwd_pd = layer_normalization_forward::primitive_desc(
      layer_norm_forward_desc, engine);

  dnnl::prop_kind p_kind = dnnl::prop_kind::backward;

  auto lnorm_bwd_d = layer_normalization_backward::desc(
      p_kind, grad_output_md, input_md, epsilon, flags);
  auto lnorm_bwd_pd = layer_normalization_backward::primitive_desc(
      lnorm_bwd_d, engine, lnorm_fwd_pd);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_memory);

  auto grad_input_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_memory);

  auto mean_memory = memory(lnorm_bwd_pd.mean_desc(), engine);
  dpcpp_set_mkldnn_buffer(mean.data_ptr(), mean_memory);

  auto var_memory = memory(lnorm_bwd_pd.variance_desc(), engine);
  dpcpp_set_mkldnn_buffer(rstd.data_ptr(), var_memory);

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, input_usr_memory},
      {DNNL_ARG_DIFF_DST, grad_output_memory},
      {DNNL_ARG_MEAN, mean_memory},
      {DNNL_ARG_VARIANCE, var_memory},
      {DNNL_ARG_DIFF_SRC, grad_input_memory},
  };
  Tensor grad_weight_bias;
  if (useScaleShift) {
    auto weight_bias_memory = memory(lnorm_bwd_pd.weights_desc(), engine);
    auto weight_bias = at::empty(2 * ih, weight.options());
    dpcppMemcpyAsync(
        weight_bias.data_ptr(),
        weight.data_ptr(),
        ih * sizeof(float),
        DeviceToDevice);
    dpcppMemsetAsync(
        static_cast<uint8_t*>(weight_bias.data_ptr()) + ih * sizeof(float),
        0,
        ih * sizeof(float));
    dpcpp_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);

    auto grad_weight_bias_memory =
        memory(lnorm_bwd_pd.diff_weights_desc(), engine);
    grad_weight_bias = at::empty(2 * ih, weight.options());
    dpcpp_set_mkldnn_buffer(
        grad_weight_bias.data_ptr(), grad_weight_bias_memory);

    args.insert({DNNL_ARG_SCALE_SHIFT, weight_bias_memory});
    args.insert({DNNL_ARG_DIFF_SCALE_SHIFT, grad_weight_bias_memory});
  }

  auto strm = GpuStreamManager::Instance().get_stream();
  layer_normalization_backward(lnorm_bwd_pd).execute(strm, args);

  if (useScaleShift) {
    dpcppMemcpyAsync(
        grad_weight.data_ptr(),
        grad_weight_bias.data_ptr(),
        ih * sizeof(float),
        DeviceToDevice);
    dpcppMemcpyAsync(
        grad_bias.data_ptr(),
        static_cast<uint8_t*>(grad_weight_bias.data_ptr()) + ih * sizeof(float),
        ih * sizeof(float),
        DeviceToDevice);
  }

  return std::make_tuple(grad_input.view(X.sizes()), grad_weight, grad_bias);
}
} // namespace AtenIpexTypeDPCPP
} // namespace at
