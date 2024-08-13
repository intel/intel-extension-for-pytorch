#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include <utils/SimpleTrace.h>
#include "Norm.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace torch::autograd;
using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor> rms_norm_fw(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& weight,
    double epsilon);

std::tuple<Tensor, Tensor> rms_norm_bw(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& rstd,
    const Tensor& weight,
    std::array<bool, 2> grad_input_mask);

template <typename scalar_t, typename mean_t, typename weight_t>
class RMSNormForward : public NormForward<scalar_t, mean_t, weight_t, true> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, mean_t, weight_t, true> NF;
  RMSNormForward() = delete;
  RMSNormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t eps,
      int64_t M,
      int64_t N)
      : NormForward<scalar_t, mean_t, weight_t, true>(
            X_data,
            Y_data,
            nullptr,
            var_data,
            gamma_data,
            nullptr,
            eps),
        M(M),
        N(N) {
    numel = M * N;
  };

  template <
      int vec_size,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename nd_item_id>
  void reduce_combine(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t& sum_value,
      accscalar_t& sum_tmp) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    for (index_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t value = *(
            reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          sum_value += Numerics<accscalar_t>::pow(value[v], 2);
        }
      }
    }
  }

  template <typename nd_item_id>
  void reduce_project(
      nd_item_id item_id,
      accscalar_t sum_value,
      accscalar_t sum_tmp,
      const NormConfig& cfg) const {
    auto group_id = item_id.get_group(0);
    accscalar_t scale = static_cast<accscalar_t>(cfg.Plane);
    NF::var_data[group_id] = static_cast<mean_t>(Numerics<accscalar_t>::rsqrt(
        sum_value < 0 ? 0
                      : sum_value / scale + static_cast<accscalar_t>(NF::eps)));
  }

  template <
      int vec_size,
      typename index_t,
      typename vec_t,
      typename weight_vec_t,
      typename nd_item_id>
  void update(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t sum_value = 0,
      accscalar_t sum_tmp = 0) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);

    index_t group_offset = group_id * cfg.Plane;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        reduce_project(item_id, sum_value, sum_tmp, cfg);
      }
      item_id.barrier(dpcpp_global_fence);
    }

    auto var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        vec_t Y_val;
        weight_vec_t gamma_val =
            *(reinterpret_cast<weight_vec_t*>(NF::gamma_data + plane_offset));

        for (int v = 0; v < vec_size; ++v) {
          Y_val[v] = static_cast<scalar_t>(gamma_val[v] * var_val * X_val[v]);
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + group_offset + plane_offset)) =
            Y_val;
      }
    }
  };

  int64_t M;
  int64_t N;
  int64_t numel;
};

template <typename scalar_t, typename mean_t, typename weight_t>
class RMSNormBackward
    : public NormBackward<scalar_t, weight_t, weight_t, true> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  RMSNormBackward() = delete;
  RMSNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* var_data,
      weight_t* gamma_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t, true>(
            X_data,
            dY_data,
            dX_data,
            nullptr,
            var_data,
            gamma_data,
            nullptr,
            nullptr),
        M(M),
        N(N) {
    numel = M * N;
  }

  RMSNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t* a_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t, true>(
            X_data,
            dY_data,
            dX_data,
            nullptr,
            var_data,
            gamma_data,
            a_data,
            nullptr),
        M(M),
        N(N) {}
  typedef NormBackward<scalar_t, mean_t, weight_t, true> NB;

  template <
      int vec_size,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename nd_item_id>
  void reduce_combine(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t& sum_value,
      accscalar_t& sum_tmp) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    mean_t rstd_val = NB::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NB::gamma_data + plane_offset));
        }
        vec_t dY_val = *(reinterpret_cast<vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NB::X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          accscalar_t value = (NB::gamma_data == nullptr)
              ? static_cast<accscalar_t>(dY_val[v])
              : (static_cast<accscalar_t>(dY_val[v]) *
                 static_cast<accscalar_t>(gamma_val[v]));
          sum_value += value * static_cast<accscalar_t>(X_val[v]) * rstd_val;
        }
      }
    }
  };

  template <
      int vec_size,
      typename index_t,
      typename vec_t,
      typename weight_vec_t,
      typename nd_item_id>
  void update(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t sum_value = 0,
      accscalar_t sum_tmp = 0) const {
    auto local_id = item_id.get_local_id(2);
    auto group_id_foreach = item_id.get_group(1);
    auto group_id = item_id.get_group(0);
    if (cfg.workgroup_num_foreach > 1) {
      sum_value = NB::a_data[group_id];
    }

    index_t group_offset = group_id * cfg.Plane;
    mean_t var_val = NB::var_data[group_id];

    int fH = cfg.Plane;
    accscalar_t term1 = (accscalar_t(1) / fH) * var_val;
    for (index_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t dY_val = *(reinterpret_cast<vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NB::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NB::gamma_data + plane_offset));
        }

        vec_t dX_val;
        for (int v = 0; v < vec_size; ++v) {
          accscalar_t f_grad_input = (NB::gamma_data == nullptr)
              ? static_cast<accscalar_t>(fH * dY_val[v])
              : static_cast<accscalar_t>(fH * gamma_val[v] * dY_val[v]);
          f_grad_input -= X_val[v] * var_val * sum_value;
          dX_val[v] = static_cast<scalar_t>(f_grad_input * term1);
        }
        *(reinterpret_cast<vec_t*>(NB::dX_data + group_offset + plane_offset)) =
            dX_val;
      }
    }
  };

  int64_t M;
  int64_t N;
  int64_t numel;
};

template <typename scalar_t, typename mean_t, typename weight_t>
void RMSNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);

  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  RMSNormForward<scalar_t, mean_t, weight_t> rms_norm_forward(
      X_data, Y_data, var_data, gamma_data, eps, M, N);
  // TODO: force it to use fused_norm_kernel
  config.workgroup_num_foreach = 1;
  config.WGPlane = config.Plane;

  if (config.workgroup_num_foreach == 1) {
    launch_vectorized_fused_norm_kernel<
        scalar_t,
        mean_t,
        weight_t,
        RMSNormForward,
        true>(rms_norm_forward, config, can_use_32bit_index);
  } else {
    Tensor semaphores, scratchpad;
    config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
    RowwiseMomentsDPCPPKernelImpl<
        scalar_t,
        mean_t,
        weight_t,
        RMSNormForward,
        true>(rms_norm_forward, config, can_use_32bit_index);
    NormUpdateKernelImpl<scalar_t, mean_t, weight_t, RMSNormForward, true>(
        rms_norm_forward, config, can_use_32bit_index);
  }
}

void RMSNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& rstd) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "RMSNormKernelImpl",
      [&]() {
        if (gamma.scalar_type() == kFloat) {
          rstd = at::empty({M}, X.options().dtype(kFloat));
          RMSNormKernelImplInternal<scalar_t, float, float>(
              X, gamma, M, N, static_cast<acc_type<scalar_t>>(eps), Y, rstd);
        } else {
          rstd = at::empty({M}, X.options());
          RMSNormKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              X, gamma, M, N, static_cast<acc_type<scalar_t>>(eps), Y, rstd);
        }
      });
}

std::tuple<Tensor, Tensor> rms_norm_fw(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& weight,
    double epsilon) {
  RECORD_FUNCTION("ipex::rms_norm_fw", std::vector<c10::IValue>({input}));

  auto M_N =
      _check_layer_norm_inputs(input, normalized_shape, weight, Tensor());
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor output = at::empty(input.sizes(), input.options());
  Tensor rstd;
  if (input.numel() != 0) {
    Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
    Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
    Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

    input_ = to_plain_if_needed(input_).contiguous();
    weight_ = to_plain_if_needed(weight_).contiguous();
    RMSNormKernelImpl(input_, weight_, M, N, epsilon, output, rstd);
  }
  return std::make_tuple(output.reshape(input.sizes()), rstd);
}

template <typename scalar_t, typename mean_t, typename weight_t>
void RmsNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    const Tensor& output,
    std::array<bool, 2> grad_input_mask) {
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(rstd.numel() == M);

  using accscalar_t = acc_type<scalar_t>;
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;

  if (grad_input_mask[0]) {
    // backward data
    scalar_t* X_data = X.data_ptr<scalar_t>();
    scalar_t* dY_data = dY.data_ptr<scalar_t>();
    scalar_t* dX_data = dX.data_ptr<scalar_t>();

    auto config = NormConfig(M, N, 1, sizeof(scalar_t));
    bool can_use_32bit_index = canUse32BitIndexMath(X) &&
        canUse32BitIndexMath(dY) && canUse32BitIndexMath(dX);

    // TODO: force it to use fused_norm_kernel
    config.workgroup_num_foreach = 1;
    config.WGPlane = config.Plane;

    if (config.workgroup_num_foreach == 1) {
      RMSNormBackward<scalar_t, mean_t, weight_t> rms_norm_backward(
          X_data, dY_data, dX_data, var_data, gamma_data, M, N);
      launch_vectorized_fused_norm_kernel<
          scalar_t,
          mean_t,
          weight_t,
          RMSNormBackward,
          true>(rms_norm_backward, config, can_use_32bit_index);
    } else {
      const auto kAccType =
          (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
          ? kFloat
          : X.scalar_type();
      Tensor a = at::empty({M}, X.options().dtype(kAccType));
      accscalar_t* a_data = a.data_ptr<accscalar_t>();

      RMSNormBackward<scalar_t, mean_t, weight_t> rms_norm_backward(
          X_data, dY_data, dX_data, var_data, gamma_data, a_data, M, N);
      Tensor semaphores, scratchpad;
      config.template init_global_reduce<accscalar_t>(
          X, semaphores, scratchpad);
      RowwiseMomentsDPCPPKernelImpl<
          scalar_t,
          mean_t,
          weight_t,
          RMSNormBackward,
          true>(rms_norm_backward, config, can_use_32bit_index);
      NormUpdateKernelImpl<scalar_t, mean_t, weight_t, RMSNormBackward, true>(
          rms_norm_backward, config, can_use_32bit_index);
    }
  }

  if (grad_input_mask[1]) {
    // backward weight
    Tensor sum_tmp = at::mul(output, dY);
    at::sum_out(dgamma, sum_tmp, at::IntArrayRef{0, 1});
  }
}

void RmsNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    const Tensor& output,
    std::array<bool, 2> grad_input_mask) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "RmsNormBackwardKernelImpl",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        if (gamma.scalar_type() == kFloat) {
          RmsNormBackwardKernelImplInternal<scalar_t, float, float>(
              dY, X, rstd, gamma, M, N, dX, dgamma, output, grad_input_mask);
        } else {
          RmsNormBackwardKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              dY, X, rstd, gamma, M, N, dX, dgamma, output, grad_input_mask);
        }
      });
}

std::tuple<Tensor, Tensor> rms_norm_bw(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& rstd,
    const Tensor& weight,
    const Tensor& output,
    std::array<bool, 2> grad_input_mask) {
  RECORD_FUNCTION("ipex::rms_norm_bw", std::vector<c10::IValue>({grad_output}));
  auto M_N =
      _check_layer_norm_inputs(input, normalized_shape, weight, Tensor());
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor grad_input;
  Tensor grad_weight;

  if (grad_input_mask[0]) {
    grad_input = at::native::empty_like(
        input,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (grad_input_mask[1]) {
    grad_weight = M > 0 ? at::native::empty_like(
                              weight,
                              c10::nullopt /* dtype */,
                              c10::nullopt /* layout */,
                              c10::nullopt /* device */,
                              c10::nullopt /* pin_memory */,
                              LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                        : at::native::zeros_like(
                              weight,
                              c10::nullopt /* dtype */,
                              c10::nullopt /* layout */,
                              c10::nullopt /* device */,
                              c10::nullopt /* pin_memory */,
                              LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (input.numel() != 0 && grad_output.numel() != 0) {
    Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
    Tensor grad_output_ =
        (grad_output.dim() == 1) ? grad_output.reshape({M, N}) : grad_output;
    Tensor weight_ =
        (weight.defined() && weight.dim() == 1) ? weight.reshape({N}) : weight;
    Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;

    input_ = input_.contiguous();
    grad_output_ = grad_output_.contiguous();
    output_ = output_.contiguous();
    weight_ = weight_.defined() ? weight_.contiguous() : weight_;

    RmsNormBackwardKernelImpl(
        grad_output_,
        input_,
        rstd,
        weight_,
        M,
        N,
        grad_input,
        grad_weight,
        output_,
        grad_input_mask);
  }
  return std::make_tuple(
      grad_input_mask[0] ? grad_input.reshape(input.sizes()) : grad_input,
      grad_input_mask[1] ? grad_weight.reshape(weight.sizes()) : grad_weight);
}

class IPEXRmsNormOp : public Function<IPEXRmsNormOp> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      const Tensor& input,
      at::IntArrayRef normalized_shape,
      const Tensor& weight,
      double epsilon) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "IPEXRmsNormOp forward -> at::AtenIpexTypeXPU::IPEXRmsNormOp::forward");
#endif
    ctx->saved_data["input_requires_grad"] = input.requires_grad();
    ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
    ctx->saved_data["normalized_shape"] = normalized_shape;
    auto outputs = rms_norm_fw(input, normalized_shape, weight, epsilon);

    ctx->save_for_backward(
        {input, weight, std::get<0>(outputs), std::get<1>(outputs)});
    variable_list result = {std::get<0>(outputs), std::get<1>(outputs)};
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "IPEXRmsNormOp backward -> at::AtenIpexTypeXPU::IPEXRmsNormOp::backward");
#endif
    auto weight_requires_grad =
        ctx->saved_data["weight_requires_grad"].toBool();
    auto input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();
    auto saved = ctx->get_saved_variables();
    Tensor input = saved[0];
    Tensor weight = saved[1];
    Tensor output = saved[2];
    Tensor rstd = saved[3];
    auto normalized_shape = weight.sizes();

    auto grad_inputs = rms_norm_bw(
        grad_outputs[0],
        input,
        normalized_shape,
        rstd,
        weight,
        output,
        {input_requires_grad, weight_requires_grad});
    return {
        std::get<0>(grad_inputs), Tensor(), std::get<1>(grad_inputs), Tensor()};
  }
};

Tensor rms_norm_impl(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& weight,
    double epsilon) {
  auto output = IPEXRmsNormOp::apply(input, normalized_shape, weight, epsilon);
  return output[0];
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "rms_norm_impl",
      at::AtenIpexTypeXPU::rms_norm_impl,
      c10::DispatchKey::AutogradXPU);
  IPEX_OP_REGISTER("rms_norm.xpu", at::AtenIpexTypeXPU::rms_norm_fw);
}
} // namespace
