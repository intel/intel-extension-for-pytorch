#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include "Norm.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {

inline std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  return std::make_pair(M, N);
}

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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "RMSNormKernelImpl",
      [&]() {
        rstd = at::empty({M}, X.options().dtype(kFloat));
        if (gamma.scalar_type() == kFloat) {
          RMSNormKernelImplInternal<scalar_t, float, float>(
              X, gamma, M, N, static_cast<acc_type<scalar_t>>(eps), Y, rstd);
        } else {
          RMSNormKernelImplInternal<scalar_t, float, scalar_t>(
              X, gamma, M, N, static_cast<acc_type<scalar_t>>(eps), Y, rstd);
        }
      });
}

std::tuple<Tensor, Tensor> rms_norm_fw(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& weight) {
  float epsilon = 1e-6;
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

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("rms_norm.xpu", at::AtenIpexTypeXPU::rms_norm_fw);
}
} // namespace
