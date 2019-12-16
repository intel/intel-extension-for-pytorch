#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED() && !AT_SYCL_ENABLED()

namespace at { namespace native {
// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon) {
    throw std::runtime_error("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, bool training,
    double epsilon, std::array<bool,3> grad_input_mask) {
    throw std::runtime_error("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

}}  // namespace at::native

#else // AT_MKLDNN_EBABLED || AT_SYCL_ENABLED

#include <c10/dpcpp/SYCLUtils.h>
#include <ATen/dpcpp/Runtime.h>
#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLMath.h>
using namespace mkldnn;
namespace at { namespace native {


std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm (
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon)
{

  auto output = at::empty(input.sizes(), input.options());

#if AT_SYCL_ENABLED()
  Device curDevice = Device(kSYCL, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

  auto propagation = training ? prop_kind::forward_training: prop_kind::forward_scoring;
  auto flag = training ? normalization_flags::use_scale_shift :
      (normalization_flags::use_scale_shift |
      normalization_flags::use_global_stats);

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);
  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {n, ic, ih, iw};
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);

  batch_normalization_forward::desc batch_norm_forward_desc(propagation, input_md, epsilon, flag);
  auto batch_norm_forward_pd  = batch_normalization_forward::primitive_desc(batch_norm_forward_desc, engine);

#if AT_SYCL_ENABLED()
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto weight_bias_memory = memory(batch_norm_forward_pd.weights_desc(), engine);
  auto weight_bias = at::empty(2*ic, weight.options());
  c10::sycl::syclMemcpyAsync(weight_bias.data_ptr(), weight.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
  c10::sycl::syclMemcpyAsync(static_cast<uint8_t*>(weight_bias.data_ptr()) + ic * sizeof(float), bias.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
  sycl_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);
  Tensor save_mean, save_var;

  auto output_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

#else


#endif
  std::shared_ptr<mkldnn::primitive> bn_fwd;
  auto strm = GpuStreamManager::Instance().get_stream();

  auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
  auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);
  bn_fwd.reset(new batch_normalization_forward(batch_norm_forward_pd));
  std::unordered_map<int, memory> args = {
    {MKLDNN_ARG_SRC, input_usr_memory},
    {MKLDNN_ARG_DST, output_usr_memory},
  };

  if ((bool)(flag & normalization_flags::use_scale_shift) &&
      mkldnn_use_scaleshift)
    args.insert({MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory});

  if ((bool)(flag & normalization_flags::use_global_stats)) {
    sycl_set_mkldnn_buffer(running_mean.data_ptr(), mean_memory);
    sycl_set_mkldnn_buffer(running_var.data_ptr(), var_memory);
  } else {
    save_mean = at::empty(ic, weight.options());
    save_var = at::empty(ic, weight.options());
    sycl_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);
    sycl_set_mkldnn_buffer(save_var.data_ptr(), var_memory);
  }

  if ((bool)(flag & normalization_flags::use_global_stats) || training) {
    args.insert({MKLDNN_ARG_MEAN, mean_memory});
    args.insert({MKLDNN_ARG_VARIANCE, var_memory});
  }

  bn_fwd->execute(strm, args);

  if (training && running_mean.defined() && running_var.defined()) {
    c10::sycl::syclMemcpyAsync(running_mean.data_ptr(), save_mean.data_ptr(), ic*sizeof(float), c10::sycl::DeviceToDevice);
    size_t  orig_size = n * ih * iw;
    size_t adjust_size = orig_size - 1;
    float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
#if AT_SYCL_ENABLED()
    c10::sycl::syclMemoryScale(running_mean.data_ptr(), save_mean.data_ptr(), ic, adjust_factor);
#else
#endif
  }

  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, bool training,
    double epsilon, std::array<bool,3> grad_input_mask)
{
  auto grad_input = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty(weight.sizes(), weight.options());
  auto grad_bias = at::empty(weight.sizes(), weight.options());

#if AT_SYCL_ENABLED()
  Device curDevice = Device(kSYCL, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

  auto flags = normalization_flags::use_scale_shift; // backward only support training mode

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw =  input.size(3);

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {n, ic, ih, iw};
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto grad_output_md = input_md;
  batch_normalization_forward::desc batch_norm_forward_desc(prop_kind::forward_training, input_md,
                                                            epsilon, flags);
  auto  batch_norm_forward_pd = batch_normalization_forward::primitive_desc(batch_norm_forward_desc, engine);

  auto bwd_desc = batch_normalization_backward::desc(prop_kind::backward, grad_output_md, input_md, epsilon, flags);
  auto bn_bwd_pd = batch_normalization_backward::primitive_desc(bwd_desc, engine, batch_norm_forward_pd);

#if AT_SYCL_ENABLED()
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_memory);

  auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
  sycl_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);

  auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);
  sycl_set_mkldnn_buffer(save_var.data_ptr(), var_memory);

  auto weight_bias_memory = memory(batch_norm_forward_pd.weights_desc(), engine);
  auto weight_bias = at::empty(2*ic, weight.options());
  c10::sycl::syclMemcpyAsync(weight_bias.data_ptr(), weight.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
  c10::sycl::syclMemsetAsync(static_cast<uint8_t*>(weight_bias.data_ptr()) + ic * sizeof(float), 0, ic * sizeof(float));
  sycl_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);

  auto grad_weight_bias_memory = memory(bn_bwd_pd.diff_weights_desc(), engine);
  auto grad_weight_bias = at::empty(2*ic, weight.options());
  sycl_set_mkldnn_buffer(grad_weight_bias.data_ptr(), grad_weight_bias_memory);

  auto grad_input_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_memory);
#else
#endif
  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<mkldnn::primitive> bn_bwd;

  bn_bwd.reset(new batch_normalization_backward(bn_bwd_pd));
  bn_bwd->execute(strm, {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_DIFF_DST, grad_output_memory},
      {MKLDNN_ARG_MEAN, mean_memory},
      {MKLDNN_ARG_VARIANCE, var_memory},
      {MKLDNN_ARG_DIFF_SRC, grad_input_memory},
      {MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory},
      {MKLDNN_ARG_DIFF_SCALE_SHIFT, grad_weight_bias_memory}});

  c10::sycl::syclMemcpyAsync(grad_weight.data_ptr(), grad_weight_bias.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
  c10::sycl::syclMemcpyAsync(grad_bias.data_ptr(), static_cast<uint8_t*>(grad_weight_bias.data_ptr()) + ic * sizeof(float), ic * sizeof(float), c10::sycl::DeviceToDevice);
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}} // at::native
#endif
