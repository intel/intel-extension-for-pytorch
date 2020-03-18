#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>

#include <core/ApplyUtils.h>
#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <core/Runtime.h>
#include <core/TensorImplUtils.h>

#include <utils/Numerics.h>

#include <ATen/aten_ipex_type_dpcpp.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename T> struct TensorMSEOp {
  void operator()(T &out, T &in1, T &in2) const {
    out = (in1 - in2) * (in1 - in2);
  }
};

template <typename T> struct TensorSubOp {
  void operator()(T &out, T &in1, T &in2) const { out = in1 - in2; }
};

template <typename T> inline T safe_log(T a) {
  if (a == 0.) {
    return Numerics<T>::log((T)1e-12);
  }
  return Numerics<T>::log(a);
}

template <typename T> struct TensorLog1Op {
  void operator()(T &out, T &in) const { out = safe_log(in); }
};

template <typename T> struct TensorLog2Op {
  void operator()(T &out, T &in) const { out = safe_log(1 - in); }
};

template <typename T> struct TensorSub2Op {
  void operator()(T &out, T &in) const { out = 1 - in; }
};

template <typename T> struct TensorBCEOp {
  void operator()(T &out, T &in, T &tar) const {
    out = -(safe_log(in) * tar + safe_log((T)1. - in) * ((T)1. - tar));
  }
};

template <typename T> struct TensorBCEGradOp {
  void operator()(T &gradInput, T &in, T &tar) const {
    gradInput = -(tar - in) / (((T)1. - in + (T)1e-12) * (in + (T)1e-12));
  }
};

template <typename T> struct TensorBCEWeightsOp {
  void operator()(T &gradInput, T &weights, T &gradOutput) const {
    gradInput = gradInput * weights * gradOutput;
  }
};

int check_size(const Tensor &t1, const Tensor &t2) {
  int d;
  if (t1.dim() != t2.dim())
    return 0;
  for (d = 0; d < t1.dim(); ++d) {
    if (t1.size(d) != t2.size(d))
      return 0;
  }
  return 1;
}

template <typename scalar_t>
void dnnl_inner_product_forward_frame(int K, scalar_t *input_data,
                                      scalar_t *target_data,
                                      scalar_t *output_data) {
  at::Device curDevice = at::Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t ic = K;
  memory::data_type data_t;
  if (std::is_same<scalar_t, at::Half>::value) {
    data_t = memory::data_type::f16;
  } else {
    data_t = memory::data_type::f32;
  }
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
  ipFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, input_md,
                                                   weight_md, output_md));
  auto ip_forward_pd =
      inner_product_forward::primitive_desc(*ipFwd_desc, engine);

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nc}, engine});
  dpcpp_set_mkldnn_buffer(input_data, input_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_oi}, engine});
  dpcpp_set_mkldnn_buffer(target_data, weight_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nc}, engine});
  dpcpp_set_mkldnn_buffer(output_data, output_usr_memory);

  std::shared_ptr<inner_product_forward> ip_forward;
  std::shared_ptr<memory> bias_usr_memory;

  bias_usr_memory.reset(new memory({{{}, data_t, format_x}, engine}));

  ip_forward.reset(new inner_product_forward(ip_forward_pd));
  ip_forward->execute(strm, {{MKLDNN_ARG_SRC, input_usr_memory},
                             {MKLDNN_ARG_WEIGHTS, weight_usr_memory},
                             {MKLDNN_ARG_BIAS, *bias_usr_memory},
                             {MKLDNN_ARG_DST, output_usr_memory}});
}

void BCECriterion_updateOutput(Tensor &output, const Tensor &input,
                               const Tensor &target, const Tensor &weights,
                               int64_t reduction) {
  TORCH_CHECK(input.numel() == target.numel(),
              "input and target have different number of elements");
  if (weights.defined()) {
    TORCH_CHECK(input.numel() == weights.numel(),
                "input and weights have different number of elements");
  }

  if (reduction == at::Reduction::None) {
    output.resize_as_(input);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "binary_cross_entropy_loss", [&] {
          at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
              output, input, target, TensorBCEOp<scalar_t>());
        });

    if (weights.defined()) {
      output.mul_(weights);
    }
    return;
  }

  output.resize_({});
  int size = input.numel();

  if (weights.defined()) {
    Tensor t0 = at::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "binary_cross_entropy_loss", [&] {
          at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
              t0, input, target, TensorBCEOp<scalar_t>());
        });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "dnnl_inner_product_forward_frame", [&] {
          scalar_t *t0_data = t0.data_ptr<scalar_t>();
          scalar_t *weights_data = weights.data_ptr<scalar_t>();
          scalar_t *output_data = output.data_ptr<scalar_t>();

          dnnl_inner_product_forward_frame(size, t0_data, weights_data,
                                           output_data);
        });
  } else {
    Tensor t1 = at::empty_like(input);
    Tensor t2 = at::empty_like(input);
    Tensor o1 = at::zeros_like(output);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "tensor_log1", [&] {
          at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
              t1, input, TensorLog1Op<scalar_t>());
        });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        t1.scalar_type(), "dnnl_inner_product_forward_frame", [&] {
          scalar_t *t1_data = t1.data_ptr<scalar_t>();
          scalar_t *target_data = target.data_ptr<scalar_t>();
          scalar_t *o1_data = o1.data_ptr<scalar_t>();

          dnnl_inner_product_forward_frame(size, t1_data, target_data, o1_data);
        });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "tensor_log2", [&] {
          at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
              t1, input, TensorLog2Op<scalar_t>());
        });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "tensor_sub2", [&] {
          at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
              t2, target, TensorSub2Op<scalar_t>());
        });
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        t1.scalar_type(), "dnnl_inner_product_forward_frame", [&] {
          scalar_t *t1_data = t1.data_ptr<scalar_t>();
          scalar_t *t2_data = t2.data_ptr<scalar_t>();
          scalar_t *output_data = output.data_ptr<scalar_t>();

          dnnl_inner_product_forward_frame(size, t1_data, t2_data, output_data);
        });
    output.mul_(-1).sub_(o1);
  }

  if (reduction == at::Reduction::Mean) {
    output.div_(size);
  }
}

void BCECriterion_updateGradInput(Tensor &gradInput, const Tensor &gradOutput,
                                  const Tensor &input, const Tensor &target,
                                  const Tensor &weights, int64_t reduction) {

  TORCH_CHECK(input.numel() == target.numel(),
              "input and target have different number of elements");
  if (weights.defined()) {
    TORCH_CHECK(input.numel() == weights.numel(),
                "input and weights have different number of elements");
  }
  gradInput.resize_as_(input);

  if (reduction == Reduction::None) {
    TORCH_CHECK(input.numel() == gradOutput.numel(),
                "input and gradOutput have different number of elements");
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "binary_cross_entropy_loss_backward", [&] {
          at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
              gradInput, input, target, TensorBCEGradOp<scalar_t>());
        });

    if (weights.defined()) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "binary_cross_entropy_loss_backward_weights",
          [&] {
            at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
                gradInput, weights, gradOutput, TensorBCEWeightsOp<scalar_t>());
          });
    } else {
      gradInput.mul_(gradOutput);
    }
    return;
  }

  TORCH_CHECK(gradOutput.dim() <= 1 && gradOutput.numel() == 1,
              "Expected a single element grad_output tensor, but got: ",
              gradOutput.sizes());

  Tensor norm = at::empty_like(gradOutput);
  norm.copy_(gradOutput);
  norm.mul_(reduction == Reduction::Mean ? 1. / ((int)input.numel()) : 1.);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "binary_cross_entropy_loss_backward", [&] {
        at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
            gradInput, input, target, TensorBCEGradOp<scalar_t>());
      });
  gradInput.mul_(norm);

  if (weights.defined())
    gradInput.mul_(weights);
}

void MSECriterion_updateOutput(Tensor &output, const Tensor &input,
                               const Tensor &target, int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");

  if (reduction != at::Reduction::None) {

    output.resize_({});
    int size = input.numel();
    Tensor in = at::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "tensor_sub", [&] {
      at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
          in, input, target, TensorSubOp<scalar_t>());
    });

    in.resize_({size});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "dnnl_inner_product_forward_frame", [&] {
          scalar_t *in_data = in.data_ptr<scalar_t>();
          scalar_t *output_data = output.data_ptr<scalar_t>();

          dnnl_inner_product_forward_frame(size, in_data, in_data, output_data);
        });

    if (reduction == at::Reduction::Mean) {
      output.div_(size);
    }
    return;
  }

  output.resize_as_(input);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "mse_loss", [&] {
    at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        output, input, target, TensorMSEOp<scalar_t>());
  });
}

} // namespace impl

Tensor &binary_cross_entropy_out(Tensor &out, const Tensor &self,
                                 const Tensor &target, const Tensor &weight,
                                 int64_t reduction) {
  impl::BCECriterion_updateOutput(out, self, target, weight, reduction);
  return out;
}

Tensor binary_cross_entropy(const Tensor &self, const Tensor &target,
                            const Tensor &weight, int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  impl::BCECriterion_updateOutput(out, self, target, weight, reduction);
  return out;
}

Tensor &
binary_cross_entropy_backward_out(Tensor &grad_input, const Tensor &grad_output,
                                  const Tensor &self, const Tensor &target,
                                  const Tensor &weight, int64_t reduction) {
  impl::BCECriterion_updateGradInput(grad_input, grad_output, self, target,
                                     weight, reduction);
  return grad_input;
}

Tensor binary_cross_entropy_backward(const Tensor &grad_output,
                                     const Tensor &self, const Tensor &target,
                                     const Tensor &weight, int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  impl::BCECriterion_updateGradInput(grad_input, grad_output, self, target,
                                     weight, reduction);
  return grad_input;
}

Tensor &mse_loss_out(Tensor &out, const Tensor &self, const Tensor &target,
                     int64_t reduction) {
  impl::MSECriterion_updateOutput(out, self, target, reduction);
  return out;
}

Tensor mse_loss(const Tensor &self, const Tensor &target, int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  impl::MSECriterion_updateOutput(out, self, target, reduction);
  return out;
}
} // namespace AtenIpexTypeDPCPP
} // namespace at
