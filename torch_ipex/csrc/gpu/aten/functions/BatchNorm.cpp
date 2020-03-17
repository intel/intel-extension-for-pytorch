#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <core/Runtime.h>
#include <core/Memory.h>
#include <utils/Math.h>


using namespace mkldnn;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_template (
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
        bool training, double momentum, double epsilon)
{
    auto output = at::empty_like(input);

    Device curDevice = Device(kDPCPP, c10::sycl::current_device());
    auto engine = GpuEngineManager::Instance().get_engine(curDevice);

    auto propagation = training ? prop_kind::forward_training: prop_kind::forward_inference;
    normalization_flags flag = normalization_flags::use_scale_shift;

    if (!weight.defined() || !bias.defined()) {
        flag &= ~normalization_flags::use_scale_shift;
//        if (mkldnn_use_scaleshift)
//            flag |= normalization_flags::use_scale_shift;
//        else
//            TODO: Add warning
    }

    if (!training) {
        if (running_mean.defined() && running_var.defined())
            // Only inference and the running_xxx is defined.
            flag |= normalization_flags::use_global_stats;
        //else
        //     May add warning. e.g: The layer_norm op is handled here.
    }

    auto input_sizes = input.sizes();
    auto input_ndim = input_sizes.size();

    int32_t n;
    int32_t ic;
    int32_t ih;
    int32_t iw;

    if (input_ndim == 3) {
        n = input.size(0);
        ic = input.size(1);
        ih = input.size(2);
        iw = 1 /*input.size(3)*/;
    } else if (input_ndim == 4) {
        n = input.size(0);
        ic = input.size(1);
        ih = input.size(2);
        iw = input.size(3);
    } else {
        std::stringstream ss;
        ss << "SYCL batch_norm backend got shape=" << input_sizes
           << ", expected input with rank 3 [n, c, h*w] or rank 4 [n, c, h, w]shape ";
        AT_ERROR(ss.str());
    }

    auto data_t = dt_to_dnnl(input.type().scalarType());
    auto format_nchw = memory::format_tag::nchw;

    memory::dims input_tz = {n, ic, ih, iw};
    auto input_md = memory::desc({input_tz}, data_t, format_nchw);

    batch_normalization_forward::desc batch_norm_forward_desc(propagation, input_md, epsilon, flag);
    auto batch_norm_forward_pd  = batch_normalization_forward::primitive_desc(batch_norm_forward_desc, engine);

    auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
    sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

    auto output_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
    sycl_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

    std::shared_ptr<mkldnn::primitive> bn_fwd;
    auto strm = GpuStreamManager::Instance().get_stream();

    bn_fwd.reset(new batch_normalization_forward(batch_norm_forward_pd));
    std::unordered_map<int, memory> args = {
            {MKLDNN_ARG_SRC, input_usr_memory},
            {MKLDNN_ARG_DST, output_usr_memory},
    };

    if ((bool)(flag & normalization_flags::use_scale_shift)) {
        auto weight_bias_memory = memory(batch_norm_forward_pd.weights_desc(), engine);
        auto weight_bias = at::empty(2*ic, weight.options().dtype(ScalarType::Float));
        Tensor _weight = weight.to(ScalarType::Float);
        Tensor _bias = bias.to(ScalarType::Float);

        if (mkldnn_use_scaleshift) {
            c10::sycl::syclMemcpyAsync(weight_bias.data_ptr(), _weight.data_ptr(), ic * sizeof(float),
                    c10::sycl::DeviceToDevice);
            c10::sycl::syclMemcpyAsync(static_cast<uint8_t*>(weight_bias.data_ptr()) + ic * sizeof(float),
                    _bias.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
            sycl_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);
        }

        args.insert({MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory});
    } else {
        //TODO: Add warning
    }

    Tensor save_mean = at::empty({ic}, input.options());
    Tensor save_var = at::empty({ic}, input.options());

    auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
    auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);

    if ((bool)(flag & normalization_flags::use_global_stats)) {
        sycl_set_mkldnn_buffer(running_mean.data_ptr(), mean_memory);
        sycl_set_mkldnn_buffer(running_var.data_ptr(), var_memory);
    }
    else {
        sycl_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);
        sycl_set_mkldnn_buffer(save_var.data_ptr(), var_memory);
    }

    args.insert({MKLDNN_ARG_MEAN, mean_memory});
    args.insert({MKLDNN_ARG_VARIANCE, var_memory});

    bn_fwd->execute(strm, args);

    if (training && running_mean.defined() && running_var.defined()) {
        c10::sycl::syclMemoryScale1(running_mean.data_ptr(), save_mean.data_ptr(), ic, momentum);
        size_t  orig_size = n * ih * iw;
        size_t adjust_size = orig_size - 1;
        float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
        c10::sycl::syclMemoryScale2(running_var.data_ptr(), save_var.data_ptr(), ic, adjust_factor, momentum);
    }

    return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm (
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double momentum, double epsilon)
{
    checkBackend("batch_norm", {input, weight, bias, running_mean, running_var}, Backend::DPCPP);

    if (input.scalar_type() != at::ScalarType::Float && input.scalar_type() != at::ScalarType::Half) {
        //TODO: add more scalar type support.
        std::stringstream ss;
        ss << "SYCL batch_norm backend got unsupported type=" << input.scalar_type();
        AT_ERROR(ss.str());
    }
    else
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "batch_norm", [&] {
            return impl::batch_norm_template<scalar_t>(
                input, weight, bias, running_mean, running_var, training, momentum, epsilon);
        });
}


std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, bool training,
    double epsilon, std::array<bool,3> grad_input_mask)
{

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (grad_input_mask[0]) {
    grad_input = at::empty_like(input);
  }
  if (grad_input_mask[1]) {
    grad_weight = at::empty_like(weight);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty_like(weight);
  }

  Device curDevice = Device(kDPCPP, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto flags = normalization_flags::use_scale_shift; // backward only support training mode

  if (!(grad_input_mask[1] && grad_input_mask[2])) {
    flags &= ~normalization_flags::use_scale_shift; // No grad_weight and grad_bias
//        if (mkldnn_use_scaleshift)
//            flag |= normalization_flags::use_scale_shift;
//        else
//            TODO: Add warning
  }

  auto input_sizes = input.sizes();
  auto input_ndim = input_sizes.size();

  int32_t n;
  int32_t ic;
  int32_t ih;
  int32_t iw;

  if (input_ndim == 3) {
    n = input.size(0);
    ic = input.size(1);
    ih = input.size(2);
    iw = 1 /*input.size(3)*/;
  } else if (input_ndim == 4) {
    n = input.size(0);
    ic = input.size(1);
    ih = input.size(2);
    iw = input.size(3);
  } else {
    std::stringstream ss;
    ss << "SYCL batch_norm backend got shape=" << input_sizes
       << ", expected input with rank 3 [n, c, h*w] or rank 4 [n, c, h, w]shape ";
    AT_ERROR(ss.str());
  }

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {n, ic, ih, iw};
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto grad_output_md = input_md;
  batch_normalization_forward::desc batch_norm_forward_desc(prop_kind::forward_training, input_md,
                                                            epsilon, flags);
  auto  batch_norm_forward_pd = batch_normalization_forward::primitive_desc(batch_norm_forward_desc, engine);

  prop_kind p_kind;

  //  The check is not nnecessarybecause the use_scale_shift defined as:
  // If not specified:
  //  - on backward propagation
  //    prop_kind == #dnnl::prop_kind::backward_data has the
  //    same behavior as prop_kind == #dnnl::prop_kind::backward

  // But the dnnl wrapper returns the output number without checking the flags:
  //  virtual int n_outputs() const override {
  //          return 1 + (desc_.prop_kind == prop_kind::backward);
  //  }
  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    p_kind = prop_kind::backward;
  }
  else {
    p_kind = prop_kind::backward_data;
  }

  auto bwd_desc = batch_normalization_backward::desc(p_kind, grad_output_md, input_md, epsilon, flags);
  auto bn_bwd_pd = batch_normalization_backward::primitive_desc(bwd_desc, engine, batch_norm_forward_pd);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_memory);

  auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
  sycl_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);

  auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);
  sycl_set_mkldnn_buffer(save_var.data_ptr(), var_memory);

  auto grad_input_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_memory);

  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<mkldnn::primitive> bn_bwd;

  bn_bwd.reset(new batch_normalization_backward(bn_bwd_pd));

  std::unordered_map<int, memory> args = {
          {MKLDNN_ARG_SRC, input_usr_memory},
          {MKLDNN_ARG_DIFF_DST, grad_output_memory},
          {MKLDNN_ARG_MEAN, mean_memory},
          {MKLDNN_ARG_VARIANCE, var_memory},
          {MKLDNN_ARG_DIFF_SRC, grad_input_memory},
  };

  Tensor grad_weight_bias;

  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    auto weight_bias_memory = memory(batch_norm_forward_pd.weights_desc(), engine);
    auto weight_bias = at::empty(2*ic, weight.options());
    c10::sycl::syclMemcpyAsync(weight_bias.data_ptr(), weight.data_ptr(), ic * sizeof(float), c10::sycl::DeviceToDevice);
    c10::sycl::syclMemsetAsync(static_cast<uint8_t*>(weight_bias.data_ptr()) + ic * sizeof(float), 0, ic * sizeof(float));
    sycl_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);

    auto grad_weight_bias_memory = memory(bn_bwd_pd.diff_weights_desc(), engine);
    grad_weight_bias = at::empty(2*ic, weight.options());
    sycl_set_mkldnn_buffer(grad_weight_bias.data_ptr(), grad_weight_bias_memory);

    args.insert({MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory});
    args.insert({MKLDNN_ARG_DIFF_SCALE_SHIFT, grad_weight_bias_memory});
  }

  bn_bwd->execute(strm, args);

  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    c10::sycl::syclMemcpyAsync(grad_weight.data_ptr(), grad_weight_bias.data_ptr(), ic * sizeof(float),
                               c10::sycl::DeviceToDevice);
    c10::sycl::syclMemcpyAsync(grad_bias.data_ptr(),
                               static_cast<uint8_t *>(grad_weight_bias.data_ptr()) + ic * sizeof(float),
                               ic * sizeof(float), c10::sycl::DeviceToDevice);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at

