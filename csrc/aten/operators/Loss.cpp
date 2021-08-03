#include <ATen/ATen.h>
#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ApplyUtils.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "Loops.h"

template <typename...>
class MultiMarginCriterionUpdateOutputKernel1 {};
template <typename...>
class MultiMarginCriterionUpdateOutputKernel2 {};
template <typename...>
class MultiMarginCriterionUpdateGradInputKernel {};

template <typename...>
class MultilabelMarginCriterionUpdateOutputKernel1 {};
template <typename...>
class MultilabelMarginCriterionUpdateOutputKernel2 {};
template <typename...>
class MultilabelMarginCriterionUpdateGradInputKernel {};

class SmoothL1Backward {};

class SmoothL1Forward {};

using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename T>
struct TensorAbsOp {
  void operator()(T& out, T& in1, T& in2) const {
    T z = in1 - in2;
    out = z >= 0 ? z : -z;
  }
};

template <typename T>
struct TensorAbsGradOp {
  void operator()(T& out, T& in1, T& in2) const {
    out = ScalarConvert<int, T>::to(in1 >= in2 ? 1 : -1);
  }
};

template <typename T>
struct TensorAbsNormOp {
  TensorAbsNormOp(T norm) : norm(norm) {}
  void operator()(T& out, T& in1, T& in2) const {
    out = (in1 >= in2 ? norm : ScalarConvert<float, T>::to(-1.) * norm);
  }

  const T norm;
};

template <typename T>
struct TensorSmoothL1Op {
  void operator()(T& out, T& in1, T& in2) const {
    T z = Numerics<T>::abs(in1 - in2);
    T one = ScalarConvert<float, T>::to(1.);
    T oneHalf = ScalarConvert<float, T>::to(0.5);
    out = z < one ? oneHalf * z * z : z - oneHalf;
  }
};

template <typename T>
struct TensorSmoothL1GradOp {
  void operator()(T& out, T& in1, T& in2) const {
    T x = in1 - in2;
    T one = ScalarConvert<float, T>::to(1.);
    if (x < -one) {
      out = -one;
    } else if (x > one) {
      out = one;
    } else {
      out = x;
    }
  }
};

template <typename T>
struct TensorSmoothL1NormOp {
  TensorSmoothL1NormOp(T norm) : norm(norm) {}

  void operator()(T& out, T& in1, T& in2) const {
    T x = in1 - in2;
    T one = ScalarConvert<float, T>::to(1.);
    if (x < -one)
      out = -one * norm;
    else if (x > one)
      out = norm;
    else
      out = norm * x;
  }

  const T norm;
};

template <typename T>
struct TensorMSEOp {
  void operator()(T& out, T& in1, T& in2) const {
    out = (in1 - in2) * (in1 - in2);
  }
};

template <typename T>
struct TensorMSENormOp {
  TensorMSENormOp(T norm) : norm(norm) {}
  void operator()(T& out, T& in1, T& in2) const {
    out = norm * (in1 - in2);
  }

  const T norm;
};

template <typename T>
struct TensorMSEGradOp {
  void operator()(T& out, T& in1, T& in2) const {
    out = ScalarConvert<int, T>::to(2) * (in1 - in2);
  }
};

template <typename T>
struct TensorSubOp {
  void operator()(T& out, T& in1, T& in2) const {
    out = in1 - in2;
  }
};

template <typename T>
inline T safe_log(T a) {
  if (a == 0.) {
    return Numerics<T>::log((T)1e-12);
  }
  return Numerics<T>::log(a);
}

template <typename T>
struct TensorLog1Op {
  void operator()(T& out, T& in) const {
    out = safe_log(in);
  }
};

template <typename T>
struct TensorLog2Op {
  void operator()(T& out, T& in) const {
    out = safe_log(1 - in);
  }
};

template <typename T>
struct TensorSub2Op {
  void operator()(T& out, T& in) const {
    out = 1 - in;
  }
};

template <typename T>
struct TensorBCEOp {
  void operator()(T& out, T& in, T& tar) const {
    out = (tar - (T)(1.)) *
            Numerics<T>::max(Numerics<T>::log((T)1 - in), (T)(-100)) -
        tar * Numerics<T>::max(Numerics<T>::log(in), (T)(-100));
  }
};

template <typename T>
struct TensorBCE2Op {
  void operator()(T& out, T& in, T& tar) const {
    out = -(safe_log(in) * tar + safe_log((T)1. - in) * ((T)1. - tar));
  }
};

template <typename T>
struct TensorBCEGradOp {
  void operator()(T& gradInput, T& in, T& tar) const {
    gradInput = -(tar - in) / (((T)1. - in + (T)1e-12) * (in + (T)1e-12));
  }
};

template <typename T>
struct TensorBCEWeightsOp {
  void operator()(T& gradInput, T& weights, T& gradOutput) const {
    gradInput = gradInput * weights * gradOutput;
  }
};

template <typename T>
struct TensorSoftMarginOp {
  void operator()(T& out, T& in1, T& in2) const {
    T one = ScalarConvert<float, T>::to(1.);
    out = safe_log(one + Numerics<T>::exp(-one * in1 * in2));
  }
};

template <typename T>
struct TensorSoftMarginGradOp {
  void operator()(T& out, T& in1, T& in2) const {
    T one = ScalarConvert<float, T>::to(1.);
    T z = Numerics<T>::exp(-one * in1 * in2);
    out = -one * in2 * z / (one + z);
  }
};

template <typename T>
struct TensorSoftMarginNormOp {
  TensorSoftMarginNormOp(T norm) : norm(norm) {}

  void operator()(T& out, T& in1, T& in2) const {
    T one = ScalarConvert<float, T>::to(1.);
    T z = Numerics<T>::exp(-one * in1 * in2);
    out = -one * norm * in2 * z / (one + z);
  }

  const T norm;
};

int check_size(const Tensor& t1, const Tensor& t2) {
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
void dnnl_inner_product_forward_frame(
    int K,
    scalar_t* input_data,
    scalar_t* target_data,
    scalar_t* output_data) {
  at::Device curDevice = at::Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t ic = K;
  memory::data_type data_t;
  if (std::is_same<scalar_t, at::Half>::value) {
    data_t = memory::data_type::f16;
  } else if (std::is_same<scalar_t, at::BFloat16>::value) {
    data_t = memory::data_type::bf16;
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

  auto ipFwd_desc = inner_product_forward::desc(
      prop_kind::forward, input_md, weight_md, output_md);
  auto ip_forward_pd =
      inner_product_forward::primitive_desc(ipFwd_desc, engine);

  auto input_usr_memory =
      dpcpp_onednn_memory({input_tz, data_t, format_nc}, engine, input_data);
  auto weight_usr_memory =
      dpcpp_onednn_memory({weight_tz, data_t, format_oi}, engine, target_data);
  auto output_usr_memory =
      dpcpp_onednn_memory({output_tz, data_t, format_nc}, engine, output_data);
  auto bias_usr_memory = memory({{}, data_t, format_x}, engine);

  auto ip_forward = inner_product_forward(ip_forward_pd);
  DPCPP_ONEDNN_EXEC(
      ip_forward,
      strm,
      {{DNNL_ARG_SRC, input_usr_memory},
       {DNNL_ARG_WEIGHTS, weight_usr_memory},
       {DNNL_ARG_BIAS, bias_usr_memory},
       {DNNL_ARG_DST, output_usr_memory}});
}

template <typename scalar_t>
void BCECriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weights,
    int64_t reduction) {
  TORCH_CHECK(
      input.numel() == target.numel(),
      "input and target have different number of elements");
  if (weights.defined()) {
    TORCH_CHECK(
        input.numel() == weights.numel(),
        "input and weights have different number of elements");
  }

  if (reduction == at::Reduction::None) {
    output.resize_as_(input);

    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        output, input, target, TensorBCEOp<scalar_t>());

    if (weights.defined()) {
      output.mul_(weights);
    }
    return;
  }

  output.resize_({});
  Tensor output_ = at::empty(input.sizes(), input.options());
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      output_, input, target, TensorBCEOp<scalar_t>());
  if (weights.defined()) {
    output_.mul_(weights);
  }

  optional<ScalarType> dtype;
  if (reduction == at::Reduction::Mean) {
    at::AtenIpexTypeXPU::mean_out(output, output_, IntArrayRef{}, false, dtype);
  } else if (reduction == at::Reduction::Sum) {
    at::AtenIpexTypeXPU::sum_out(output, output_, IntArrayRef{}, false, dtype);
  }
}

template <typename scalar_t>
void BCECriterion_updateGradInput(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weights,
    int64_t reduction) {
  TORCH_CHECK(
      input.numel() == target.numel(),
      "input and target have different number of elements");
  if (weights.defined()) {
    TORCH_CHECK(
        input.numel() == weights.numel(),
        "input and weights have different number of elements");
  }
  gradInput.resize_as_(input);

  if (reduction == Reduction::None) {
    TORCH_CHECK(
        input.numel() == gradOutput.numel(),
        "input and gradOutput have different number of elements");

    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        gradInput, input, target, TensorBCEGradOp<scalar_t>());

    if (weights.defined()) {
      DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
          gradInput, weights, gradOutput, TensorBCEWeightsOp<scalar_t>());

    } else {
      gradInput.mul_(gradOutput);
    }
    return;
  }

  TORCH_CHECK(
      gradOutput.dim() <= 1 && gradOutput.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      gradOutput.sizes());

  Tensor norm = at::empty_like(gradOutput);
  norm.copy_(gradOutput);
  norm.mul_(reduction == Reduction::Mean ? 1. / ((int)input.numel()) : 1.);

  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      gradInput, input, target, TensorBCEGradOp<scalar_t>());

  gradInput.mul_(norm);

  if (weights.defined())
    gradInput.mul_(weights);
}

template <typename scalar_t>
void MSECriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");

  if (reduction != at::Reduction::None) {
    output.resize_({});
    Tensor output_ = at::empty(input.sizes(), input.options());
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        output_, input, target, TensorMSEOp<scalar_t>());
    optional<ScalarType> dtype;
    if (reduction == at::Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(
          output, output_, IntArrayRef{}, false, dtype);
    } else if (reduction == at::Reduction::Sum) {
      at::AtenIpexTypeXPU::sum_out(
          output, output_, IntArrayRef{}, false, dtype);
    }

    return;
  }

  output.resize_as_(input);

  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      output, input, target, TensorMSEOp<scalar_t>());
}

template <typename scalar_t>
void MSECriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(
      input.numel() == target.numel(),
      "input and target have different number of elements");
  grad_input.resize_as_(input);

  if (reduction == Reduction::None) {
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        grad_input, input, target, TensorMSEGradOp<scalar_t>());
    at::mul_out(grad_input, grad_input, grad_output);
    return;
  }
  scalar_t gradOutput0d = grad_output.item().to<scalar_t>();
  scalar_t norm =
      (reduction == Reduction::Mean ? 2. / (scalar_t)input.numel() : 2.) *
      gradOutput0d;
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      grad_input, input, target, TensorMSENormOp<scalar_t>(norm));
}

template <typename scalar_t>
void AbsCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");

  if (reduction == Reduction::None) {
    output.resize_as_(input);
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        output, input, target, TensorAbsOp<scalar_t>());
    return;
  }

  output.resize_({});
  Tensor output_ = at::empty(input.sizes(), input.options());
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      output_, input, target, TensorAbsOp<scalar_t>());
  optional<ScalarType> dtype;
  if (reduction == at::Reduction::Mean) {
    at::AtenIpexTypeXPU::mean_out(output, output_, IntArrayRef{}, false, dtype);
  } else if (reduction == at::Reduction::Sum) {
    at::AtenIpexTypeXPU::sum_out(output, output_, IntArrayRef{}, false, dtype);
  }
}

template <typename scalar_t>
void AbsCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");
  grad_input.resize_as_(input);

  if (reduction == Reduction::None) {
    TORCH_CHECK(
        check_size(grad_output, input),
        "input and gradOutput shape do not match");
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        grad_input, input, target, TensorAbsGradOp<scalar_t>());
    at::mul_out(grad_input, grad_input, grad_output);
    return;
  }

  scalar_t gradOutput0d = grad_output.item().to<scalar_t>();
  scalar_t norm =
      (reduction == Reduction::Mean ? 1. / (scalar_t)input.numel() : 1.) *
      gradOutput0d;
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      grad_input, input, target, TensorAbsNormOp<scalar_t>(norm));
}

void smooth_l1_kernel(TensorIterator& iter, double beta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "smooth_l1_kernel",
      [&iter, beta]() {
        scalar_t beta_val(beta);
        dpcpp_kernel_for_tensor_iter<SmoothL1Forward>(
            iter, [beta_val](scalar_t a, scalar_t b) -> scalar_t {
              auto z = ::abs(a - b);
              return z < beta_val ? scalar_t(0.5) * z * z / beta_val
                                  : z - scalar_t(0.5) * beta_val;
            });
      });
}

void smooth_l1_backward_kernel(TensorIterator& iter, Scalar norm, double beta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "smooth_l1_backward_kernel",
      [&iter, &norm, beta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t beta_val(beta);
        dpcpp_kernel_for_tensor_iter<SmoothL1Backward>(
            iter,
            [norm_val, beta_val](
                scalar_t input,
                scalar_t target,
                scalar_t grad_output) -> scalar_t {
              const auto x = input - target;
              if (x < -beta_val)
                return -norm_val * grad_output;
              else if (x > beta_val)
                return norm_val * grad_output;
              else
                return norm_val * x * grad_output / beta_val;
            });
      });
}

template <typename scalar_t>
void SoftMarginCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");

  if (reduction == Reduction::None) {
    output.resize_as_(input);
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        output, input, target, TensorSoftMarginOp<scalar_t>());
    return;
  }

  output.resize_({});
  Tensor output_ = at::empty(input.sizes(), input.options());
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      output_, input, target, TensorSoftMarginOp<scalar_t>());
  optional<ScalarType> dtype;
  if (reduction == at::Reduction::Mean) {
    at::AtenIpexTypeXPU::mean_out(output, output_, IntArrayRef{}, false, dtype);
  } else if (reduction == at::Reduction::Sum) {
    at::AtenIpexTypeXPU::sum_out(output, output_, IntArrayRef{}, false, dtype);
  }
}

template <typename scalar_t>
void SoftMarginCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  TORCH_CHECK(check_size(input, target), "input and target shape do not match");
  grad_input.resize_as_(input);

  if (reduction == Reduction::None) {
    DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        grad_input, input, target, TensorSoftMarginGradOp<scalar_t>());
    at::mul_out(grad_input, grad_input, grad_output);
    return;
  }
  scalar_t gradOutput0d = grad_output.item().to<scalar_t>();
  scalar_t norm =
      (reduction == Reduction::Mean ? 1. / (scalar_t)input.numel() : 1.) *
      gradOutput0d;
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      grad_input, input, target, TensorSoftMarginNormOp<scalar_t>(norm));
}

template <typename scalar_t>
void MultiMarginCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  const int p_ = p.toInt();
  const double margin_ = margin.toDouble();
  TORCH_CHECK(p_ == 1 || p_ == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  if (reduction == Reduction::None && target.dim() > 0) {
    output.resize_({nframe});
  } else {
    output.resize_({});
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weights_contiguous =
      (weights.defined()) ? weights.contiguous() : weights;

  bool has_weights = weights.defined() ? true : false;

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto weights_data = has_weights
        ? weights_contiguous.data_ptr<scalar_t>()
        : input_data; // use the input_data handler as dummy weights
    auto local_output_data = dpcpp_local_acc_t<scalar_t>(local_size, cgh);

    if (reduction == Reduction::None && output.dim() > 0) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto local_item_id = item_id.get_id(0);
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          auto target_idx = target_ptr[i];
          auto input_target = input_ptr[i * dim + target_idx];
          for (auto d = 0; d < dim; d++) {
            scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
            if (d == target_idx)
              continue;
            if (z > 0) {
              scalar_t h = (p_ == 1) ? z : z * z;
              if (weights_ptr)
                h *= weights_ptr[target_idx];
              sum += h;
            }
          }
          sum /= dim;
          output_ptr[i] = sum;
        }
      };
      cgh.parallel_for<MultiMarginCriterionUpdateOutputKernel1<scalar_t>>(
          DPCPP::range<1>(local_size), kfn);
    } else {
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto local_item_id = item_id.get_local_id(0);
        local_output_data[local_item_id] = 0.0;
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          auto target_idx = target_ptr[i];
          auto input_target = input_ptr[i * dim + target_idx];
          for (auto d = 0; d < dim; d++) {
            scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
            if (d == target_idx)
              continue;
            if (z > 0) {
              scalar_t h = (p_ == 1) ? z : z * z;
              if (weights_ptr)
                h *= weights_ptr[target_idx];
              sum += h;
            }
          }
          sum /= dim;
          if (reduction == Reduction::Mean)
            sum /= nframe;
          local_output_data[local_item_id] += sum;
        }

        // reduce
        for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
          item_id.barrier(dpcpp_global_and_local_fence);
          if (local_item_id < i)
            local_output_data[local_item_id] +=
                local_output_data[local_item_id + i];
        }
        item_id.barrier(dpcpp_global_and_local_fence);
        output_ptr[0] = local_output_data[0];
      };
      cgh.parallel_for<MultiMarginCriterionUpdateOutputKernel2<scalar_t>>(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(local_size), DPCPP::range<1>(local_size)),
          kfn);
    }
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void MultiMarginCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  const int p_ = p.toInt();
  const double margin_ = margin.toDouble();
  TORCH_CHECK(p_ == 1 || p_ == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weights_contiguous =
      (weights.defined()) ? weights.contiguous() : weights;

  bool has_weights = weights.defined() ? true : false;
  scalar_t g = (reduction == Reduction::Mean)
      ? static_cast<scalar_t>(1. / (nframe * dim))
      : static_cast<scalar_t>(1. / dim);

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto weights_data = has_weights
        ? weights_contiguous.data_ptr<scalar_t>()
        : input_data; // use the input_data handler as dummy weights

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto input_ptr = input_data;
      auto target_ptr = target_data;
      auto weights_ptr = has_weights ? weights_data : NULL;
      auto local_item_id = item_id.get_id(0);

      for (int i = local_item_id; i < nframe; i += local_size) {
        auto target_idx = target_ptr[i];
        auto input_target = input_ptr[i * dim + target_idx];
        scalar_t grad_input_target = 0;
        for (auto d = 0; d < dim; d++) {
          scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
          if (d == target_idx)
            continue;
          if (z > 0) {
            scalar_t h = (p_ == 1) ? g : 2 * g * z;
            if (weights_ptr)
              h *= weights_ptr[target_idx];
            grad_input_target -= h;
            grad_input_ptr[i * dim + d] = h;
          } else
            grad_input_ptr[i * dim + d] = 0;
        }
        grad_input_ptr[i * dim + target_idx] = grad_input_target;

        for (auto d = 0; d < dim; d++)
          grad_input_ptr[i * dim + d] *= reduction == Reduction::None
              ? grad_output_ptr[i]
              : grad_output_ptr[0];
      }
    };
    cgh.parallel_for<MultiMarginCriterionUpdateGradInputKernel<scalar_t>>(
        DPCPP::range<1>(local_size), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void MultilabelMarginCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& is_target) {
  auto target_arg = TensorArg(target, "target", 2);

  const auto ndims = input.dim();

  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() <= 1 && target.numel() == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  }

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, "is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  is_target.resize_as_(target);
  TORCH_CHECK(is_target.is_contiguous(), "is_target must be contiguous");
  is_target.zero_();

  if (reduction != Reduction::None || target.dim() <= 1) {
    output.resize_({});
  } else {
    output.resize_({nframe});
  }

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto is_target_data = is_target.data_ptr<scalar_t>();
    auto local_output_data = dpcpp_local_acc_t<scalar_t>(local_size, cgh);

    if (reduction == Reduction::None && output.dim() > 0) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto is_target_ptr = is_target_data;
        auto local_item_id = item_id.get_id(0);
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          for (int64_t ddt = 0; ddt < dim; ddt++) {
            auto target_idx = target_ptr[i * dim + ddt];
            if (target_idx < 0)
              break;
            is_target_ptr[i * dim + target_idx] = 1;
          }
          for (int64_t dt = 0; dt < dim; dt++) {
            auto target_idx = target_ptr[i * dim + dt];
            if (target_idx < 0)
              break;

            auto input_target = input_ptr[i * dim + target_idx];
            for (int64_t d = 0; d < dim; d++) {
              if (!is_target_ptr[i * dim + d]) {
                scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
                if (z > 0)
                  sum += z;
              }
            }
          }
          sum /= dim;
          output_ptr[i] = sum;
        }
      };
      cgh.parallel_for<MultilabelMarginCriterionUpdateOutputKernel1<scalar_t>>(
          DPCPP::range<1>(local_size), kfn);
    } else {
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto is_target_ptr = is_target_data;
        auto local_item_id = item_id.get_local_id(0);
        local_output_data[local_item_id] = 0.0;
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          for (int64_t ddt = 0; ddt < dim; ddt++) {
            auto target_idx = target_ptr[i * dim + ddt];
            if (target_idx < 0)
              break;
            is_target_ptr[i * dim + target_idx] = 1;
          }
          for (int64_t dt = 0; dt < dim; dt++) {
            auto target_idx = target_ptr[i * dim + dt];
            if (target_idx < 0)
              break;

            auto input_target = input_ptr[i * dim + target_idx];
            for (int64_t d = 0; d < dim; d++) {
              if (!is_target_ptr[i * dim + d]) {
                scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
                if (z > 0)
                  sum += z;
              }
            }
          }
          sum /= dim;
          if (reduction == Reduction::Mean)
            sum /= nframe;
          local_output_data[local_item_id] += sum;
        }

        // reduce
        for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
          item_id.barrier(dpcpp_global_and_local_fence);
          if (local_item_id < i)
            local_output_data[local_item_id] +=
                local_output_data[local_item_id + i];
        }
        item_id.barrier(dpcpp_global_and_local_fence);
        output_ptr[0] = local_output_data[0];
      };
      cgh.parallel_for<MultilabelMarginCriterionUpdateOutputKernel2<scalar_t>>(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(local_size), DPCPP::range<1>(local_size)),
          kfn);
    }
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void MultilabelMarginCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto target_arg = TensorArg(target, "target", 3);
  auto is_target_arg = TensorArg(is_target, "is_target", 5);

  const auto ndims = input.dim();

  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() <= 1 && target.numel() == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  }
  IsOnSameDevice(
      "multilabel_margin_loss_backward_out", target_arg, is_target_arg);

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, "is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto is_target_contiguous = is_target.contiguous();

  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  grad_input.zero_();

  auto is_target_cont_arg =
      TensorArg(is_target_contiguous, "is_target_cont", 5);
  TORCH_CHECK(
      is_target_contiguous.min().item<scalar_t>() >= 0,
      is_target_cont_arg,
      " is out of range");
  TORCH_CHECK(
      is_target_contiguous.max().item<scalar_t>() <= 1,
      is_target_cont_arg,
      " is out of range");

  scalar_t g = static_cast<scalar_t>(
      reduction == Reduction::Mean ? 1. / (nframe * dim) : 1. / dim);

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto is_target_data = is_target_contiguous.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto input_ptr = input_data;
      auto target_ptr = target_data;
      auto is_target_ptr = is_target_data;
      auto local_item_id = item_id.get_id(0);

      for (int i = local_item_id; i < nframe; i += local_size) {
        for (int64_t dt = 0; dt < dim; dt++) {
          auto target_idx = target_ptr[i * dim + dt];
          if (target_idx < 0)
            break;

          auto input_target = input_ptr[i * dim + target_idx];
          for (int64_t d = 0; d < dim; d++) {
            if (!is_target_ptr[i * dim + d]) {
              scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
              if (z > 0) {
                grad_input_ptr[i * dim + target_idx] -= g;
                grad_input_ptr[i * dim + d] += g;
              }
            }
          }
        }
        for (int64_t d = 0; d < dim; d++)
          grad_input_ptr[i * dim + d] *= (reduction == Reduction::None)
              ? grad_output_ptr[i]
              : grad_output_ptr[0];
      }
    };
    cgh.parallel_for<MultilabelMarginCriterionUpdateGradInputKernel<scalar_t>>(
        DPCPP::range<1>(local_size), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor& binary_cross_entropy_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bce_loss_out",
      [&] {
        impl::BCECriterion_updateOutput<scalar_t>(
            out, self, target, weight, reduction);
      });
  return out;
}

Tensor binary_cross_entropy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::binary_cross_entropy_out(
      out, self, target, weight, reduction);
}

Tensor& binary_cross_entropy_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bce_loss_backward_out",
      [&] {
        impl::BCECriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, weight, reduction);
      });
  return grad_input;
}

Tensor binary_cross_entropy_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::binary_cross_entropy_backward_out(
      grad_input, grad_output, self, target, weight, reduction);
}

Tensor& mse_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mse_loss_out",
      [&] {
        impl::MSECriterion_updateOutput<scalar_t>(out, self, target, reduction);
      });
  return out;
}

Tensor mse_loss(const Tensor& self, const Tensor& target, int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::mse_loss_out(out, self, target, reduction);
}

Tensor& mse_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mse_loss_backward_out",
      [&] {
        impl::MSECriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, reduction);
      });
  return grad_input;
}

Tensor mse_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::mse_loss_backward_out(
      grad_input, grad_output, self, target, reduction);
}

Tensor& l1_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "l1_loss_out",
      [&] {
        impl::AbsCriterion_updateOutput<scalar_t>(out, self, target, reduction);
      });
  return out;
}

Tensor l1_loss(const Tensor& self, const Tensor& target, int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::l1_loss_out(out, self, target, reduction);
}

Tensor& l1_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "l1_loss_backward_out",
      [&] {
        impl::AbsCriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, reduction);
      });
  return grad_input;
}

Tensor l1_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::l1_loss_backward_out(
      grad_input, grad_output, self, target, reduction);
}

Tensor& smooth_l1_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (beta == 0) {
    return at::AtenIpexTypeXPU::l1_loss_out(out, self, target, reduction);
  }
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, self, target);
    impl::smooth_l1_kernel(iter, beta);
    if (reduction == Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(out, iter.output(), 0, false, c10::nullopt);
    } else {
      at::AtenIpexTypeXPU::sum_out(out, iter.output(), 0, false, c10::nullopt);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    impl::smooth_l1_kernel(iter, beta);
  }
  return out;
}

static inline at::Tensor apply_loss_reduction(
    const at::Tensor& unreduced,
    int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

Tensor smooth_l1_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (beta == 0) {
    return at::AtenIpexTypeXPU::l1_loss(self, target, reduction);
  }
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::smooth_l1_kernel(iter, beta);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& smooth_l1_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  if (beta <= 0)
    return at::AtenIpexTypeXPU::l1_loss_backward_out(
        grad_input, grad_output, self, target, reduction);
  auto norm = reduction == Reduction::Mean ? 1. / self.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  impl::smooth_l1_backward_kernel(iter, norm, beta);
  return grad_input;
}

Tensor smooth_l1_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  if (beta <= 0)
    return at::AtenIpexTypeXPU::l1_loss_backward(
        grad_output, self, target, reduction);
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::smooth_l1_loss_backward_out(
      grad_input, grad_output, self, target, reduction, beta);
}

Tensor& soft_margin_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "soft_margin_loss_out",
      [&] {
        impl::SoftMarginCriterion_updateOutput<scalar_t>(
            out, self, target, reduction);
      });
  return out;
}

Tensor soft_margin_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::soft_margin_loss_out(
      out, self, target, reduction);
}

Tensor& soft_margin_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "soft_margin_loss_backward_out",
      [&] {
        impl::SoftMarginCriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, reduction);
      });
  return grad_input;
}

Tensor soft_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::soft_margin_loss_backward_out(
      grad_input, grad_output, self, target, reduction);
}

Tensor& multi_margin_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multi_margin_loss_out",
      [&] {
        impl::MultiMarginCriterion_updateOutput<scalar_t>(
            out, self, target, p, margin, weights, reduction);
      });
  return out;
}

Tensor multi_margin_loss(
    const Tensor& self,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multi_margin_loss_out(
      out, self, target, p, margin, weights, reduction);
}

Tensor& multi_margin_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multi_margin_loss_backward_out",
      [&] {
        impl::MultiMarginCriterion_updateGradInput<scalar_t>(
            grad_input,
            grad_output,
            self,
            target,
            p,
            margin,
            weights,
            reduction);
      });
  return grad_input;
}

Tensor multi_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weights,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multi_margin_loss_backward_out(
      grad_input, grad_output, self, target, p, margin, weights, reduction);
}

Tensor& multilabel_margin_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor is_target = at::empty({0}, self.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_out",
      [&] {
        impl::MultilabelMarginCriterion_updateOutput<scalar_t>(
            out, self, target, reduction, is_target);
      });
  return out;
}

Tensor multilabel_margin_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_out(
      out, self, target, reduction);
}

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out(
    Tensor& output,
    Tensor& is_target,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_forward_out",
      [&] {
        impl::MultilabelMarginCriterion_updateOutput<scalar_t>(
            output, self, target, reduction, is_target);
      });
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor output = at::empty({0}, self.options());
  Tensor is_target = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_forward_out(
      output, is_target, self, target, reduction);
}

Tensor& multilabel_margin_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_backward_out",
      [&] {
        impl::MultilabelMarginCriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, reduction, is_target);
      });
  return grad_input;
}

Tensor multilabel_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_backward_out(
      grad_input, grad_output, self, target, reduction, is_target);
}

} // namespace AtenIpexTypeXPU
} // namespace at
