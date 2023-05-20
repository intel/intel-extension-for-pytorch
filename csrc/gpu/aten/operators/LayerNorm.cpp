#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include "Norm.h"
#include "comm/RegistrationDeclarations.h"

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
class LayerNormForward : public NormForward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, mean_t, weight_t> NF;
  LayerNormForward() = delete;
  LayerNormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps,
      int64_t M,
      int64_t N)
      : NormForward<scalar_t, mean_t, weight_t>(
            X_data,
            Y_data,
            mean_data,
            var_data,
            gamma_data,
            beta_data,
            eps),
        M(M),
        N(N) {
    numel = M * N;
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
      accscalar_t sum1 = 0,
      accscalar_t sum2 = 0) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);

    index_t group_offset = group_id * cfg.Plane;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        NF::reduce_project(item_id, sum1, sum2, cfg);
      }
      item_id.barrier(dpcpp_global_fence);
    }

    mean_t mean_val = NF::mean_data[group_id];
    mean_t var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val, beta_val;
        vec_t Y_val;
        if (NF::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NF::gamma_data + plane_offset));
        }
        if (NF::beta_data != nullptr) {
          beta_val =
              *(reinterpret_cast<weight_vec_t*>(NF::beta_data + plane_offset));
        }

        for (int v = 0; v < vec_size; ++v) {
          if (NF::gamma_data != nullptr && NF::beta_data != nullptr) {
            Y_val[v] = static_cast<accscalar_t>(gamma_val[v]) *
                    (var_val * static_cast<accscalar_t>(X_val[v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else if (NF::gamma_data != nullptr) {
            Y_val[v] = static_cast<accscalar_t>(gamma_val[v]) *
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val));
          } else if (NF::beta_data != nullptr) {
            Y_val[v] =
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else {
            Y_val[v] =
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val));
          }
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
class LayerNormBackward : public NormBackward<scalar_t, weight_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  LayerNormBackward() = delete;
  LayerNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t>(
            X_data,
            dY_data,
            dX_data,
            mean_data,
            var_data,
            gamma_data,
            nullptr,
            nullptr),
        M(M),
        N(N) {
    numel = M * N;
  }

  LayerNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t* a_data,
      accscalar_t* b_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t>(
            X_data,
            dY_data,
            dX_data,
            mean_data,
            var_data,
            gamma_data,
            a_data,
            b_data),
        M(M),
        N(N) {}
  typedef NormBackward<scalar_t, mean_t, weight_t> NB;

  template <
      int vec_size,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename nd_item_id>
  void reduce_combine(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t& sum1,
      accscalar_t& sum2) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    mean_t mean_val = NB::mean_data[group_id];
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
          sum1 += value;
          sum2 +=
              value * static_cast<accscalar_t>(X_val[v] - mean_val) * rstd_val;
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
      accscalar_t sum1 = 0,
      accscalar_t sum2 = 0) const {
    auto local_id = item_id.get_local_id(2);
    auto group_id_foreach = item_id.get_group(1);
    auto group_id = item_id.get_group(0);
    if (cfg.workgroup_num_foreach > 1) {
      sum1 = NB::a_data[group_id];
      sum2 = NB::b_data[group_id];
    }

    index_t group_offset = group_id * cfg.Plane;
    mean_t mean_val = NB::mean_data[group_id];
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
          f_grad_input -= (X_val[v] - mean_val) * var_val * sum2;
          dX_val[v] = static_cast<scalar_t>((f_grad_input - sum1) * term1);
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
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  TORCH_CHECK(!beta.defined() || beta.numel() == N);

  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  LayerNormForward<scalar_t, mean_t, weight_t> layer_norm_forward(
      X_data, Y_data, mean_data, var_data, gamma_data, beta_data, eps, M, N);
  if (config.workgroup_num_foreach == 1) {
    launch_vectorized_fused_norm_kernel<
        scalar_t,
        mean_t,
        weight_t,
        LayerNormForward>(layer_norm_forward, config, can_use_32bit_index);
  } else {
    Tensor semaphores, scratchpad;
    config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
    RowwiseMomentsDPCPPKernelImpl<scalar_t, mean_t, weight_t, LayerNormForward>(
        layer_norm_forward, config, can_use_32bit_index);
    NormUpdateKernelImpl<scalar_t, mean_t, weight_t, LayerNormForward>(
        layer_norm_forward, config, can_use_32bit_index);
  }
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        if (gamma.scalar_type() == kFloat) {
          mean = at::empty({M}, X.options().dtype(kFloat));
          rstd = at::empty({M}, X.options().dtype(kFloat));
          LayerNormKernelImplInternal<scalar_t, float, float>(
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t>>(eps),
              Y,
              mean,
              rstd);
        } else {
          mean = at::empty({M}, X.options());
          rstd = at::empty({M}, X.options());
          LayerNormKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t>>(eps),
              Y,
              mean,
              rstd);
        }
      });
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    int vec_size>
void GammaBetaBackwardSimpleDPCPPKernel(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor& dgamma,
    Tensor& dbeta,
    NormConfig& cfg) {
  scalar_t* dY_data = dY.data_ptr<scalar_t>();
  scalar_t* X_data = X.data_ptr<scalar_t>();
  weight_t* dg_data = dgamma.data_ptr<weight_t>();
  weight_t* db_data = dbeta.data_ptr<weight_t>();

  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t =
      at::native::Memory::aligned_vector_loop<weight_t, vec_size>;

  sycl::range<3> local_range{1, cfg.block_row, cfg.workgroup_size};
  sycl::range<3> global_range{
      cfg.workgroup_num, cfg.block_row, cfg.workgroup_size};

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t, 3> local_sum1(
        sycl::range<3>(cfg.block_row, cfg.workgroup_size, vec_size), cgh);
    dpcpp_local_acc_t<accscalar_t, 3> local_sum2(
        sycl::range<3>(cfg.block_row, cfg.workgroup_size, vec_size), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_range), sycl::range<3>(local_range)),
        [=](sycl::nd_item<3> item_id) {
          auto local_row_id = item_id.get_local_id(1);
          auto local_col_id = item_id.get_local_id(2);
          auto group_id = item_id.get_group(0);

          accscalar_t dg_sum1[vec_size], db_sum1[vec_size];
#pragma unroll(vec_size)
          for (int v = 0; v < vec_size; ++v) {
            dg_sum1[v] = 0;
            db_sum1[v] = 0;
          }

          for (int row_id = local_row_id; row_id < cfg.Batch;
               row_id += cfg.block_row) {
            accscalar_t mean_val = mean_data[row_id];
            accscalar_t rstd_val = var_data[row_id];
            auto plane_offset =
                (group_id * cfg.workgroup_size + local_col_id) * vec_size;
            if (plane_offset < cfg.Plane) {
              auto offset = row_id * cfg.Plane + plane_offset;
              vec_t X_val = *(reinterpret_cast<vec_t*>(X_data + offset));
              vec_t dY_val = *(reinterpret_cast<vec_t*>(dY_data + offset));
#pragma unroll(vec_size)
              for (int v = 0; v < vec_size; ++v) {
                dg_sum1[v] += (dg_data == nullptr)
                    ? accscalar_t(0)
                    : static_cast<accscalar_t>(dY_val[v]) *
                        (static_cast<accscalar_t>(X_val[v]) - mean_val) *
                        rstd_val;
                db_sum1[v] += (db_data == nullptr)
                    ? accscalar_t(0)
                    : static_cast<accscalar_t>(dY_val[v]);
              }
            }
          }

          if (cfg.block_row > 1) {
            norm_group_reduce_row<vec_size, accscalar_t>(
                item_id,
                dg_sum1,
                db_sum1,
                local_sum1,
                local_sum2,
                cfg.block_row,
                [](accscalar_t a, accscalar_t b) { return a + b; });
          }

          if (local_row_id == 0) {
            auto plane_offset =
                (group_id * cfg.workgroup_size + local_col_id) * vec_size;
            if (plane_offset < cfg.Plane) {
              weight_vec_t dg_val, db_val;
              if (cfg.block_row > 1) {
#pragma unroll(vec_size)
                for (int v = 0; v < vec_size; ++v) {
                  dg_val[v] =
                      static_cast<weight_t>(local_sum1[0][local_col_id][v]);
                  db_val[v] =
                      static_cast<weight_t>(local_sum2[0][local_col_id][v]);
                }
              } else {
#pragma unroll(vec_size)
                for (int v = 0; v < vec_size; ++v) {
                  dg_val[v] = static_cast<weight_t>(dg_sum1[v]);
                  db_val[v] = static_cast<weight_t>(db_sum1[v]);
                }
              }
              *(reinterpret_cast<weight_vec_t*>(dg_data + plane_offset)) =
                  dg_val;
              *(reinterpret_cast<weight_vec_t*>(db_data + plane_offset)) =
                  db_val;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t>
void GammaBetaBackwardSimpleDPCPPKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor& dgamma,
    Tensor& dbeta,
    NormConfig& config) {
  bool can_use_32bit_index =
      canUse32BitIndexMath(X) && canUse32BitIndexMath(dY);
#define VecGammaBetaBackwardSimpleDPCPPKernel(vec_size)             \
  GammaBetaBackwardSimpleDPCPPKernel<                               \
      scalar_t,                                                     \
      accscalar_t,                                                  \
      mean_t,                                                       \
      weight_t,                                                     \
      vec_size>(dY, X, mean_data, var_data, dgamma, dbeta, config); \
  break;

  switch (config.max_vec_size) {
    case 8: {
      VecGammaBetaBackwardSimpleDPCPPKernel(8);
    }
    case 4: {
      VecGammaBetaBackwardSimpleDPCPPKernel(4);
    }
    case 2: {
      VecGammaBetaBackwardSimpleDPCPPKernel(2);
    }
    case 1: {
      VecGammaBetaBackwardSimpleDPCPPKernel(1);
    }
  }
}

template <typename scalar_t, typename mean_t, typename weight_t>
void LayerNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(mean.numel() == M);
  TORCH_CHECK(rstd.numel() == M);

  using accscalar_t = acc_type<scalar_t>;
  mean_t* mean_data = mean.data_ptr<mean_t>();
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
    if (config.workgroup_num_foreach == 1) {
      LayerNormBackward<scalar_t, mean_t, weight_t> layer_norm_backward(
          X_data, dY_data, dX_data, mean_data, var_data, gamma_data, M, N);
      launch_vectorized_fused_norm_kernel<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward>(layer_norm_backward, config, can_use_32bit_index);
    } else {
      const auto kAccType =
          (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
          ? kFloat
          : X.scalar_type();
      Tensor a = at::empty({M}, X.options().dtype(kAccType));
      Tensor b = at::empty({M}, X.options().dtype(kAccType));
      accscalar_t* a_data = a.data_ptr<accscalar_t>();
      accscalar_t* b_data = b.data_ptr<accscalar_t>();

      LayerNormBackward<scalar_t, mean_t, weight_t> layer_norm_backward(
          X_data,
          dY_data,
          dX_data,
          mean_data,
          var_data,
          gamma_data,
          a_data,
          b_data,
          M,
          N);
      Tensor semaphores, scratchpad;
      config.template init_global_reduce<accscalar_t>(
          X, semaphores, scratchpad);
      RowwiseMomentsDPCPPKernelImpl<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward>(layer_norm_backward, config, can_use_32bit_index);
      NormUpdateKernelImpl<scalar_t, mean_t, weight_t, LayerNormBackward>(
          layer_norm_backward, config, can_use_32bit_index);
    }
  }

  if (grad_input_mask[1]) {
    // backward weight
    auto config = NormConfig(M, N, 0, sizeof(scalar_t));
    GammaBetaBackwardSimpleDPCPPKernelImpl<
        scalar_t,
        accscalar_t,
        mean_t,
        weight_t>(dY, X, mean_data, var_data, dgamma, dbeta, config);
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormBackwardKernelImpl",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        if (gamma.scalar_type() == kFloat) {
          LayerNormBackwardKernelImplInternal<scalar_t, float, float>(
              dY,
              X,
              mean,
              rstd,
              gamma,
              M,
              N,
              dX,
              dgamma,
              dbeta,
              grad_input_mask);
        } else {
          LayerNormBackwardKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              dY,
              X,
              mean,
              rstd,
              gamma,
              M,
              N,
              dX,
              dgamma,
              dbeta,
              grad_input_mask);
        }
      });
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor output = at::empty(input.sizes(), input.options());
  Tensor mean, rstd;
  if (input.numel() != 0) {
    bool apply_onednn =
        (xpu::oneDNN::is_onednn_layout(input) && input.dim() > 1);
    if (apply_onednn) {
      Tensor input_ = xpu::oneDNN::contiguous_if_needed(input);
      Tensor weight_ = xpu::oneDNN::contiguous_if_needed(weight);
      Tensor bias_ = xpu::oneDNN::contiguous_if_needed(bias);
      std::tie(output, mean, rstd) =
          xpu::oneDNN::layer_norm(input_, weight_, bias_, epsilon);
    } else {
      Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
      Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
      Tensor weight_ = (weight.defined() && weight.dim() == 1)
          ? weight.reshape({N})
          : weight;
      Tensor bias_ =
          (bias.defined() && bias.dim() == 1) ? bias.reshape({N}) : bias;
      input_ = input_.contiguous();
      weight_ = weight_.defined() ? weight_.contiguous() : weight_;
      bias_ = bias_.defined() ? bias_.contiguous() : bias_;
      LayerNormKernelImpl(
          input_, weight_, bias_, M, N, epsilon, output, mean, rstd);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor grad_input = at::empty(input.sizes(), input.options());
  Tensor grad_weight, grad_bias;
  if (weight.numel() != 0) {
    grad_weight = at::zeros(weight.sizes(), weight.options());
    grad_bias = at::zeros(bias.sizes(), bias.options());
  } else {
    grad_weight = at::empty(weight.sizes(), weight.options());
    grad_bias = at::empty(bias.sizes(), bias.options());
  }
  if (input.numel() != 0 && grad_output.numel() != 0) {
    if (M > 0 && N > 0) {
      bool apply_onednn =
          (xpu::oneDNN::is_onednn_layout(input) && input.dim() > 1);
      if (apply_onednn) {
        Tensor grad_output_ = xpu::oneDNN::contiguous_if_needed(grad_output);
        Tensor input_ = xpu::oneDNN::contiguous_if_needed(input);
        Tensor weight_ = xpu::oneDNN::contiguous_if_needed(weight);
        std::tie(grad_input, grad_weight, grad_bias) =
            xpu::oneDNN::layer_norm_backward(
                grad_output_, input_, mean, rstd, weight_, 1e-5);
      } else {
        Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
        Tensor grad_input_ =
            (grad_input.dim() == 1) ? grad_input.reshape({M, N}) : grad_input;
        Tensor grad_output_ = (grad_output.dim() == 1)
            ? grad_output.reshape({M, N})
            : grad_output;
        Tensor weight_ = (weight.defined() && weight.dim() == 1)
            ? weight.reshape({N})
            : weight;

        input_ = input_.contiguous();
        grad_output_ = grad_output_.contiguous();
        weight_ = weight_.defined() ? weight_.contiguous() : weight_;

        LayerNormBackwardKernelImpl(
            grad_output_,
            input_,
            mean,
            rstd,
            weight_,
            M,
            N,
            grad_input_,
            grad_weight,
            grad_bias,
            grad_input_mask);
      }
    }
  }
  return std::make_tuple(
      grad_input.reshape(input.sizes()),
      weight.defined() ? grad_weight.reshape(weight.sizes()) : grad_weight,
      bias.defined() ? grad_bias.reshape(bias.sizes()) : grad_bias);
}

} // namespace AtenIpexTypeXPU
} // namespace at
