#include "dnnl_ops.h"
#include "MKLDNNCommon.h"
#include "Utils.h"
#include <torch/csrc/autograd/VariableTypeUtils.h>

namespace at { namespace native {

using attr_t = ideep::descriptor_group::attr_t;

at::Tensor dnnl_conv2d_base(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    attr_t attr_) {
  // transfer Varibale problem to Tensor?
  const ideep::tensor x = get_mkldnn_tensor(input);
  const ideep::tensor w = get_mkldnn_tensor(weight);
  c10::optional<ideep::tensor> b{c10::nullopt};
  if (bias.numel() != 0) {
    b = get_mkldnn_tensor(bias);
  }

  std::vector<int64_t> kernel_size(x.ndims());
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(groups > 1,
        "Only group dnnl_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(w.get_dims().cbegin() + 2, x.ndims() - 1,
                kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());          
  }

  const ideep::param::dims x_dims = x.get_dims();
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> output_sizes = conv_output_size(
      input_size, kernel_size, padding, stride, dilation);

  ideep::tensor y;
  if (b.has_value()) {
    ideep::convolution_forward::compute<alloc>(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        attr_,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  } else {
    ideep::convolution_forward::compute<alloc>(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        attr_,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  }

  return torch::autograd::make_variable(
      new_with_itensor_mkldnn(std::move(y), input.options()));
}

// TODO: fold dnnl_conv2d_base into dnnl_conv2d_base_inplace
at::Tensor& dnnl_conv2d_base_inplace(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    attr_t attr_) {
  // transfer Varibale problem to Tensor?
  const ideep::tensor x = get_mkldnn_tensor(input);
  const ideep::tensor w = get_mkldnn_tensor(weight);
  c10::optional<ideep::tensor> b{c10::nullopt};
  if (bias.numel() != 0) {
    b = get_mkldnn_tensor(bias);
  }
  auto y = get_mkldnn_tensor(output);

  std::vector<int64_t> kernel_size(x.ndims());
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(groups > 1,
        "Only group dnnl_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(w.get_dims().cbegin() + 2, x.ndims() - 1,
                kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());          
  }

  const ideep::param::dims x_dims = x.get_dims();
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> output_sizes = conv_output_size(
      input_size, kernel_size, padding, stride, dilation);

  if (b.has_value()) {
    ideep::convolution_forward::compute(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        attr_,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        attr_,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference);
  }

  return output;
}

at::Tensor dnnl_conv2d(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  return dnnl_conv2d_base(
      input, weight, bias, stride, padding, dilation, groups, attr_t());
}

at::Tensor dnnl_conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  return dnnl_conv2d_base(
      input, weight, bias, stride, padding, dilation, groups,
      attr_t::fuse_relu());
}

at::Tensor& dnnl_conv2d_sum(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Tensor& accumu, at::Scalar alpha) {
  auto scale = alpha.to<float>();
  return dnnl_conv2d_base_inplace(
      input, weight, bias, accumu, stride, padding, dilation, groups,
      attr_t::fuse_sum(scale));
}

at::Tensor& dnnl_conv2d_sum_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Tensor& accumu, at::Scalar alpha) {
  auto scale = alpha.to<float>();
  return dnnl_conv2d_base_inplace(
      input, weight, bias, accumu, stride, padding, dilation, groups,
      attr_t::residual(scale));
}

at::Tensor dnnl_reorder(
    const at::Tensor& input,
    ideep::format from, ideep::format to, int64_t groups) {
  AT_ASSERTM(input.scalar_type() == c10::ScalarType::Float,
             "dnnl_reorder: Expects float tensor input");
  AT_ASSERTM(input.dim() <= 5,
             "dnnl_reorder: Can't convert cpu tensor with dimensions > 5");

  // `get_mkldnn_tensor` accepts both aten and dnnl tensors
  auto src_itensor = get_mkldnn_tensor(input);
  if (from != ideep::format::any)
    AT_ASSERTM(src_itensor.get_descriptor().get_internal_format() == from
        || src_itensor.as_weights().get_internal_format() == from,
               "dnnl_reorder: Incompatible input format");

  if (to == ideep::format::nchw) {
    auto dims = src_itensor.get_dims();
    // casts int32_t dims to int64_t
    auto sizes = std::vector<int64_t>(dims.begin(), dims.end());
    auto cpu_tensor = at::empty(sizes, input.options().layout(c10::kStrided));
    // do reordering
    src_itensor.to_public(cpu_tensor.template data_ptr<float>());
    return cpu_tensor;
  } else {
    auto& input_cont = input.is_mkldnn() ? input : input.contiguous();
    // pre-alloc a tensor, managed by ideep
    auto dst_tensor = empty_mkldnn(input_cont.sizes(), input_cont.options(),
        to, groups);
    auto& dst_itensor = itensor_from_mkldnn(dst_tensor);
    // do reordering
    dst_itensor.feed_from(dst_itensor.get_dims(), ideep::tensor::data_type::f32,
                          input_cont.template data_ptr<float>());
    return dst_tensor;
  }
}

at::Tensor dnnl_relu(const at::Tensor& input) {
  const ideep::tensor x = get_mkldnn_tensor(input);

  // TODO: input continues check
  auto output = _empty_like(input);
  auto y = get_mkldnn_tensor(output);

  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu,
      ideep::prop_kind::forward_inference, /*alpha*/ 0.0);
  return output;
}

at::Tensor& dnnl_relu_(at::Tensor& input) {
  ideep::tensor x = get_mkldnn_tensor(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_inference, /*alpha*/ 0.0);
  return input;
}

at::Tensor dnnl_pooling_2d_base(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    ideep::algorithm algorithm) {
  const ideep::tensor self_input = get_mkldnn_tensor(self);
  const ideep::param::dims self_input_dims = self_input.get_dims();
  std::vector<int64_t> input_size{self_input_dims.cbegin(), self_input_dims.cend()};
  std::vector<int64_t> output_size;
  std::vector<int64_t> padding_vec_l, padding_vec_r;
  padding_vec_l = padding_vec_r = padding.vec();

  if (ceil_mode) {
    // dnnl does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input_size,
        kernel_size,
        padding_vec_l,
        padding_vec_r,
        stride,
        dilation,
        true);

    // adjust padding until output sizes agree
    bool all_equal = false;
    while (!all_equal) {
      output_size = pool_output_sizes(
          input_size,
          kernel_size,
          padding_vec_l,
          padding_vec_r,
          stride,
          dilation,
          false);

      all_equal = true;
      for (size_t i = 2; i < input_size.size(); ++i) {
        if (output_size[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2] ++;
           all_equal = false;
        }
      }
    }
  } else {
    output_size = pool_output_sizes(
        input_size,
        kernel_size,
        padding_vec_l,
        padding_vec_r,
        stride,
        dilation,
        false);
  }

  ideep::tensor y;
  ideep::pooling_forward::compute<alloc>(
      self_input,
      {output_size.cbegin(), output_size.cend()},
      y,
      {stride.begin(), stride.end()},
      {kernel_size.begin(), kernel_size.end()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algorithm,
      ideep::prop_kind::forward);

  return torch::autograd::make_variable(
      new_with_itensor_mkldnn(std::move(y), self.options()));
}

at::Tensor dnnl_pooling_avg_2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode) {
  return dnnl_pooling_2d_base(
      self,
      kernel_size,
      stride,
      padding,
      {1, 1},
      ceil_mode,
      ideep::algorithm::pooling_avg);
}

at::Tensor dnnl_pooling_max_2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  return dnnl_pooling_2d_base(
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

at::Tensor dnnl_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_cuda) {
  AT_ASSERTM(input.dim() == 4 || input.dim() == 5,
             "dnnl_batch_norm: currently dnnl only support 2d and 3d batchnorm");
  AT_ASSERTM(weight.numel() != 0 && bias.numel() !=0,
             "dnnl_batch_norm: currently dnnl only support affine model");

  const ideep::tensor x = get_mkldnn_tensor(input);
  const ideep::tensor w = get_mkldnn_tensor(weight);
  const ideep::tensor b = get_mkldnn_tensor(bias);
  bool use_running_stat = (running_mean.numel()!=0  && running_var.numel() !=0);

  auto output = _empty_like(input);
  auto y = get_mkldnn_tensor(output);

  if (train) {
    // perharps this path not be used now
    ideep::tensor saved_mean;
    ideep::tensor saved_var;
    if (use_running_stat) {
      ideep::tensor m = get_mkldnn_tensor(running_mean);
      ideep::tensor v = get_mkldnn_tensor(running_var);
      ideep::batch_normalization_forward_training::compute(
          x, w, b, y, saved_mean, saved_var, m, v, momentum, eps);
    } else {
      ideep::batch_normalization_forward_training::compute(
          x, w, b, y, saved_mean, saved_var, momentum, eps);
    }
  } else {
    if (use_running_stat) {
      ideep::tensor m = get_mkldnn_tensor(running_mean);
      ideep::tensor v = get_mkldnn_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      ideep::batch_normalization_forward_inference::compute(
          x, w, b, y, eps);
    }
  }
  return output;
}

at::Tensor dnnl_fold_weight(const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps) {
  // dnnl conv2d weights could have been re-ordered to 5d
  auto out_channels = weight.ndimension() == 5 ? weight.size(0) * weight.size(1) : weight.size(0);
  at::Tensor w_conv, w, var;
  if (weight.is_mkldnn()) {
    w_conv = weight.to_dense().view({out_channels, -1});
    if (bn_weight.numel() != 0) {
      w =  bn_weight.is_mkldnn() ? bn_weight.to_dense() : bn_weight;
    } else {
      w = at::ones(out_channels, w_conv.options());
    }
    if (running_var.numel() != 0) {
      var = var.is_mkldnn() ? running_var.to_dense() : running_var;
    } else {
      var = at::ones(out_channels, w_conv.options());
    }
  } else {
    w_conv = weight.view({out_channels, -1});
    if (bn_weight.numel() != 0) {
      w =  bn_weight.is_mkldnn() ? bn_weight.to_dense() : bn_weight;
    } else {
      w = at::ones(out_channels, w_conv.options());
    }
    if (running_var.numel() != 0) {
      var = running_var.is_mkldnn() ? running_var.to_dense() : running_var;
      var = at::ones(out_channels, w_conv.options());
    }
  }

  auto w_bn = at::diag(w.div(at::sqrt(eps + var)));
  auto dst_dense = at::mm(w_bn, w_conv).view(weight.sizes());

  if (weight.is_mkldnn()) {
    // reorder to format same as weight
    ideep::tensor weight_ideep = itensor_from_mkldnn(weight);
    ideep::tensor y;
    y.init<alloc>(weight_ideep.get_descriptor());
    ideep::tensor dst_ideep = itensor_view_from_dense(dst_dense);
    y.feed_from(dst_ideep);
    return torch::autograd::make_variable(
      new_with_itensor_mkldnn(std::move(y), weight.options()));
  } else {
    return dst_dense;
  }
}

at::Tensor dnnl_fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps) {
  auto out_channels = bias.numel() == 0 ? bn_weight.numel() == 0
    ? bn_bias.numel() == 0 ? running_mean.numel() == 0 ? running_var.numel() == 0 ? 0 :
    running_var.size(0) : running_mean.size(0) : bn_bias.size(0) : bn_weight.size(0) : bias.size(0);

  if (out_channels == 0)
    return bias;

  auto options = bias.numel() == 0 ? bn_weight.numel() == 0
    ? bn_bias.numel() == 0 ? running_mean.numel() == 0 ? running_var.numel() == 0 ?
    c10::TensorOptions() : running_var.options() : running_mean.options()
    : bn_bias.options() : bn_weight.options() : bias.options();

  auto conv_bias = bias.numel() == 0 ? at::zeros(out_channels, options)
    : bias.is_mkldnn() ? bias.to_dense() : bias;
  auto bn_w = bn_weight.numel() == 0 ? at::ones(out_channels, options)
    : bn_weight.is_mkldnn() ? bn_weight.to_dense() : bn_weight;
  auto bn_b = bn_bias.numel() == 0 ? at::zeros(out_channels, options)
    : bn_bias.is_mkldnn()? bn_bias.to_dense() :bn_bias;
  auto mean = running_mean.numel() == 0 ? at::zeros(out_channels, options)
    : running_mean.is_mkldnn() ? running_mean.to_dense() : running_mean;
  auto var = running_var.numel() == 0 ? at::ones(out_channels, options)
    : running_var.is_mkldnn() ? running_var.to_dense() : running_var;

  auto dst_dense = bn_b + bn_w.div(at::sqrt(var + eps)).mul(conv_bias - mean);
  return dst_dense;
}

at::Tensor dnnl_sum(
    const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  std::vector<ideep::tensor> inputs {
    get_mkldnn_tensor(self), get_mkldnn_tensor(other)
  };

  auto output = _empty_like(self);
  auto y = get_mkldnn_tensor(output);
  const std::vector<float> scales {1.0, alpha.to<float>()};
  ideep::sum::compute(scales, inputs, y);

  return output;
}

at::Tensor& dnnl_sum_(
    at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  // Correct?
  auto self_ = get_mkldnn_tensor(self);
  std::vector<ideep::tensor> inputs{
    self_, get_mkldnn_tensor(other)
  };
  const std::vector<float> scales {1.0, alpha.to<float>()};
  ideep::sum::compute(scales, inputs, self_);

  return self;
}
}} // namespace torch::jit
