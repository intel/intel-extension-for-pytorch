#include "Norm.h"

using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename mean_t, typename weight_t>
class GroupNormForward : public NormForward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  GroupNormForward() = delete;
  GroupNormForward(
      scalar_t* X_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps,
      int64_t N,
      int64_t C,
      int64_t group,
      int64_t HxW)
      : NormForward<scalar_t, mean_t, weight_t>(
            X_data,
            nullptr,
            mean_data,
            var_data,
            gamma_data,
            beta_data,
            eps),
        N(N),
        C(C),
        group(group),
        HxW(HxW) {
    numel = N * C * HxW;
    D = C / group;
  }
  typedef NormForward<scalar_t, mean_t, weight_t> NF;

  void set_eltwise_update_parameter(
      scalar_t* X_ptr,
      scalar_t* Y_ptr,
      accscalar_t* a_ptr,
      accscalar_t* b_ptr,
      bool is_channels_last) {
    NF::X_data = X_ptr;
    NF::Y_data = Y_ptr;
    a_data = a_ptr;
    b_data = b_ptr;
    channels_last = is_channels_last;
  }

  template <int vec_size, typename index_t, typename vec_t>
  void eltwise_update(index_t i) const {
    index_t remaining = numel - i * vec_size;
    if (remaining < vec_size) {
      for (int j = 0; j < remaining; ++j) {
        index_t offset = i * vec_size + j;

        index_t nc;
        if (channels_last) {
          nc = offset / (C * HxW) * C + offset % C;
        } else {
          nc = offset / HxW;
        }
        NF::Y_data[offset] = static_cast<scalar_t>(
            a_data[nc] * static_cast<accscalar_t>(NF::X_data[offset]) +
            b_data[nc]);
      }
    } else {
      index_t offset = i * vec_size;

      vec_t in_val = *(reinterpret_cast<vec_t*>(NF::X_data + offset));
      vec_t out_val;
#pragma unroll(vec_size)
      for (int v = 0; v < vec_size; ++v) {
        index_t nc;
        if (channels_last) {
          nc = (offset + v) / (C * HxW) * C + (offset + v) % C;
        } else {
          nc = (offset + v) / HxW;
        }
        out_val[v] = static_cast<scalar_t>(
            a_data[nc] * static_cast<accscalar_t>(in_val[v]) + b_data[nc]);
      }
      *(reinterpret_cast<vec_t*>(NF::Y_data + offset)) = out_val;
    }
  };

  int N;
  int C;
  int group;
  int HxW;
  int D;
  int numel;
  accscalar_t* a_data;
  accscalar_t* b_data;
  bool channels_last;
};

template <typename scalar_t, typename mean_t, typename weight_t>
class GroupNormBackward : public NormBackward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  GroupNormBackward() = delete;
  GroupNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      accscalar_t* a_data,
      accscalar_t* b_data,
      int64_t N,
      int64_t C,
      int64_t group,
      int64_t HxW)
      : NormBackward<scalar_t, mean_t, weight_t>(
            X_data,
            dY_data,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            a_data,
            b_data),
        N(N),
        C(C),
        group(group),
        HxW(HxW) {
    numel = N * C * HxW;
    D = C / group;
  }

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

    for (int64_t j = local_id * vec_size; j < cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t dY_val = *(reinterpret_cast<vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NB::X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          sum1 += static_cast<accscalar_t>(dY_val[v]) *
              static_cast<accscalar_t>(X_val[v]);
          sum2 += static_cast<accscalar_t>(dY_val[v]);
        }
      }
    }
  }

  void set_eltwise_update_parameter(
      accscalar_t* c1_ptr,
      accscalar_t* c2_ptr,
      accscalar_t* c3_ptr,
      scalar_t* X_ptr,
      scalar_t* dY_ptr,
      scalar_t* dX_ptr,
      bool is_channels_last) {
    c1_data = c1_ptr;
    c2_data = c2_ptr;
    c3_data = c3_ptr;
    NB::X_data = X_ptr;
    NB::dY_data = dY_ptr;
    NB::dX_data = dX_ptr;
    channels_last = is_channels_last;
  }

  template <int vec_size, typename index_t, typename vec_t>
  void eltwise_update(index_t i) const {
    index_t remaining = numel - i * vec_size;
    if (remaining < vec_size) {
      for (int j = 0; j < remaining; ++j) {
        index_t offset = i * vec_size + j;

        index_t nc, ng;
        if (channels_last) {
          nc = offset / (C * HxW) * C + offset % C;
          ng = nc / D;
        } else {
          nc = offset / HxW;
          ng = nc / D;
        }
        NB::dX_data[offset] =
            c1_data[nc] * static_cast<accscalar_t>(NB::dY_data[offset]) +
            c2_data[ng] * static_cast<accscalar_t>(NB::X_data[offset]) +
            c3_data[ng];
      }
    } else {
      index_t offset = i * vec_size;
      vec_t X_val = *(reinterpret_cast<vec_t*>(NB::X_data + offset));
      vec_t dY_val = *(reinterpret_cast<vec_t*>(NB::dY_data + offset));
      vec_t dX_val;
#pragma unroll(vec_size)
      for (int v = 0; v < vec_size; ++v) {
        index_t nc, ng;
        if (channels_last) {
          nc = (offset + v) / (C * HxW) * C + (offset + v) % C;
          ng = nc / D;
        } else {
          nc = (offset + v) / HxW;
          ng = nc / D;
        }
        dX_val[v] = c1_data[nc] * static_cast<accscalar_t>(dY_val[v]) +
            c2_data[ng] * static_cast<accscalar_t>(X_val[v]) + c3_data[ng];
      }
      *(reinterpret_cast<vec_t*>(NB::dX_data + offset)) = dX_val;
    }
  }

  int N;
  int C;
  int group;
  int HxW;
  int D;
  int numel;
  accscalar_t* c1_data;
  accscalar_t* c2_data;
  accscalar_t* c3_data;
  bool channels_last;
};

template <typename scalar_t, typename mean_t, typename weight_t>
void ComputeFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const mean_t* mean_data,
    const mean_t* rstd_data,
    const weight_t* gamma_data,
    const weight_t* beta_data,
    acc_type<scalar_t>* a_data,
    acc_type<scalar_t>* b_data) {
  using accscalar_t = acc_type<scalar_t>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto global_range = N * C;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>(global_range), [=](sycl::item<1> item_id) {
      auto id = item_id.get_id(0);

      const int64_t ng = id / (C / group);
      const int64_t c = id % C;
      const accscalar_t x = (gamma_data == nullptr)
          ? static_cast<accscalar_t>(rstd_data[ng])
          : static_cast<accscalar_t>(rstd_data[ng]) *
              static_cast<accscalar_t>(gamma_data[c]);
      a_data[id] = x;
      b_data[id] = -x * static_cast<accscalar_t>(mean_data[ng]) +
          (beta_data == nullptr ? accscalar_t(0)
                                : static_cast<accscalar_t>(beta_data[c]));
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename mean_t, typename weight_t>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    scalar_t eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  using accscalar_t = acc_type<scalar_t>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  Tensor X_cont = X.contiguous();
  scalar_t* X_data = X_cont.data_ptr<scalar_t>();

  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* rstd_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(N * group, C / group * HxW, 1, sizeof(scalar_t));
  GroupNormForward<scalar_t, mean_t, weight_t> group_norm_forward(
      X_data,
      mean_data,
      rstd_data,
      gamma_data,
      beta_data,
      eps,
      N,
      C,
      group,
      HxW);
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  Tensor semaphores, scratchpad;
  config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
  RowwiseMomentsDPCPPKernelImpl<scalar_t, mean_t, weight_t, GroupNormForward>(
      group_norm_forward, config, can_use_32bit_index);

  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  accscalar_t* a_data = a.data_ptr<accscalar_t>();
  accscalar_t* b_data = b.data_ptr<accscalar_t>();
  ComputeFusedParamsDPCPPKernel<scalar_t, mean_t, weight_t>(
      N, C, group, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);

  // propagate channels_last format from X to Y
  bool is_channels_last = is_smf_channels_last(X);
  if (is_channels_last) {
    X_cont = X;
    auto smf = X.suggest_memory_format();
    Y = at::empty_like(X, smf);
  } else {
    Y = at::empty_like(X_cont);
  }
  group_norm_forward.set_eltwise_update_parameter(
      X_cont.data_ptr<scalar_t>(),
      Y.data_ptr<scalar_t>(),
      a_data,
      b_data,
      is_channels_last);
  NormEltwiseUpdateKernelImpl<scalar_t, mean_t, weight_t, GroupNormForward>(
      group_norm_forward, config, can_use_32bit_index);
}

template <typename accscalar_t, typename mean_t, typename weight_t>
void ComputeGradOutputCoeffientDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const mean_t* rstd,
    const weight_t* gamma,
    accscalar_t* c1) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = N * C;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> item_id) {
      auto nc = item_id.get_id(0);

      const int64_t ng = nc / (C / group);
      const int64_t c = nc % C;
      c1[nc] = static_cast<accscalar_t>(rstd[ng]) *
          (gamma == nullptr ? accscalar_t(1)
                            : static_cast<accscalar_t>(gamma[c]));
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename accscalar_t, typename mean_t, typename weight_t>
void ComputeBackwardFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const mean_t* mean,
    const mean_t* rstd,
    const weight_t* gamma,
    const accscalar_t* ds,
    const accscalar_t* db,
    accscalar_t* c2,
    accscalar_t* c3) {
  // mean, rstd [N, C] -> [N][G][D]
  // gamma [C] -> [G][D]
  // ds, db [N, C] -> [N][G][D]
  // c2, c3 [N, G] -> [N][G]
  auto G = group;
  auto D = C / G;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  workgroup_size = std::min(workgroup_size, D);
  int sub_group_num = (workgroup_size + SIMD - 1) / SIMD;
  workgroup_size = sub_group_num * SIMD;
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(N, group * workgroup_size),
            sycl::range<2>(1, workgroup_size)),
        [=](sycl::nd_item<2> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          auto local_id = item_id.get_local_id(1);
          auto n = item_id.get_group(0);
          auto g = item_id.get_group(1);
          auto ng = n * G + g;
          accscalar_t sum1 = 0;
          accscalar_t sum2 = 0;
          for (int64_t i = local_id; i < D; i += workgroup_size) {
            auto nc = ng * D + i;
            auto c = g * D + i;
            const accscalar_t gamma_v = gamma == nullptr
                ? accscalar_t(1)
                : static_cast<accscalar_t>(gamma[c]);
            sum1 += ds[nc] * gamma_v;
            sum2 += db[nc] * gamma_v;
          }

          norm_group_reduce<accscalar_t>(
              item_id,
              sub_group_num,
              sum1,
              sum2,
              local_sum1,
              local_sum2,
              [](accscalar_t a, accscalar_t b) { return a + b; });

          if (local_id == 0) {
            const accscalar_t s =
                accscalar_t(1) / static_cast<accscalar_t>(D * HxW);
            const accscalar_t x =
                (sum2 * static_cast<accscalar_t>(mean[ng]) - sum1) *
                static_cast<accscalar_t>(rstd[ng]) *
                static_cast<accscalar_t>(rstd[ng]) *
                static_cast<accscalar_t>(rstd[ng]) * s;
            c2[ng] = x;
            c3[ng] = -x * static_cast<accscalar_t>(mean[ng]) -
                sum2 * static_cast<accscalar_t>(rstd[ng]) * s;
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename mean_t, typename weight_t>
void GammaBetaBackwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const mean_t* mean,
    const mean_t* rstd,
    const acc_type<scalar_t>* ds,
    const acc_type<scalar_t>* db,
    weight_t* dgamma,
    weight_t* dbeta) {
  // mean, rstd: {N, group}
  // ds, db: {N, C}  {N, group, D}
  // dgamma, dbeta: {C}  {group, D}
  using accscalar_t = acc_type<scalar_t>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  auto total_threads =
      ((C + workgroup_size - 1) / workgroup_size) * workgroup_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_threads), sycl::range<1>(workgroup_size)),
        [=](sycl::nd_item<1> item_id) {
          auto index = item_id.get_global_linear_id();
          if (index < C) {
            auto G = group;
            auto D = C / G;
            accscalar_t sum1 = 0;
            accscalar_t sum2 = 0;
            for (int64_t n = 0; n < N; ++n) {
              auto nc = n * C + index;
              auto ng = n * G + index / D;
              sum1 += (dgamma == nullptr)
                  ? accscalar_t(0)
                  : ((ds[nc] - db[nc] * static_cast<accscalar_t>(mean[ng])) *
                     static_cast<accscalar_t>(rstd[ng]));
              sum2 += (dbeta == nullptr) ? accscalar_t(0) : db[nc];
            }
            if (dgamma != nullptr) {
              dgamma[index] = sum1;
            }
            if (dbeta != nullptr) {
              dbeta[index] = sum2;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename mean_t, typename weight_t>
void GroupNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  using accscalar_t = acc_type<scalar_t>;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * group);
  TORCH_CHECK(rstd.numel() == N * group);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  if (N == 0) {
    return;
  }

  Tensor X_cont = X.contiguous();
  Tensor dY_cont = dY.contiguous();
  const mean_t* mean_data = mean.data_ptr<mean_t>();
  const mean_t* rstd_data = rstd.data_ptr<mean_t>();
  const weight_t* gamma_data =
      gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;

  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  accscalar_t* ds_data = ds.data_ptr<accscalar_t>();
  accscalar_t* db_data = db.data_ptr<accscalar_t>();

  bool can_use_32bit_index =
      canUse32BitIndexMath(X_cont) && canUse32BitIndexMath(dY_cont);
  auto config = NormConfig(N * C, HxW, 1, sizeof(scalar_t));
  GroupNormBackward<scalar_t, mean_t, weight_t> group_norm_backward(
      X_cont.data_ptr<scalar_t>(),
      dY_cont.data_ptr<scalar_t>(),
      ds_data,
      db_data,
      N,
      C,
      group,
      HxW);
  Tensor semaphores, scratchpad;
  config.template init_global_reduce<accscalar_t>(X, semaphores, scratchpad);
  RowwiseMomentsDPCPPKernelImpl<scalar_t, mean_t, weight_t, GroupNormBackward>(
      group_norm_backward, config, can_use_32bit_index);

  // compute gradient input (dX)
  if (grad_input_mask[0]) {
    Tensor c1 = at::empty({N, C}, X.options().dtype(kAccType));
    accscalar_t* c1_data = c1.data_ptr<accscalar_t>();
    ComputeGradOutputCoeffientDPCPPKernel<accscalar_t, mean_t, weight_t>(
        N, C, group, rstd_data, gamma_data, c1_data);

    Tensor c2 = at::empty({N, group}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, group}, X.options().dtype(kAccType));
    accscalar_t* c2_data = c2.data_ptr<accscalar_t>();
    accscalar_t* c3_data = c3.data_ptr<accscalar_t>();
    ComputeBackwardFusedParamsDPCPPKernel<accscalar_t, mean_t, weight_t>(
        N,
        C,
        HxW,
        group,
        mean_data,
        rstd_data,
        gamma_data,
        ds_data,
        db_data,
        c2_data,
        c3_data);

    bool is_channels_last = is_smf_channels_last(X);
    if (is_channels_last) {
      auto smf = X.suggest_memory_format();
      dX = at::empty_like(X, smf);
      X_cont = X;
      dY_cont = dY.contiguous(smf);
    } else {
      dX = at::empty_like(X_cont);
    }
    group_norm_backward.set_eltwise_update_parameter(
        c1_data,
        c2_data,
        c3_data,
        X_cont.data_ptr<scalar_t>(),
        dY_cont.data_ptr<scalar_t>(),
        dX.data_ptr<scalar_t>(),
        is_channels_last);
    NormEltwiseUpdateKernelImpl<scalar_t, mean_t, weight_t, GroupNormBackward>(
        group_norm_backward, config, can_use_32bit_index);
  }

  // compute gradient weight (dgamma and dbeta)
  if (grad_input_mask[1] || grad_input_mask[2]) {
    weight_t* dgamma_data = nullptr;
    if (grad_input_mask[1]) {
      dgamma = at::empty_like(gamma);
      dgamma_data = dgamma.data_ptr<weight_t>();
    }
    weight_t* dbeta_data = nullptr;
    if (grad_input_mask[2]) {
      dbeta = at::empty_like(gamma);
      dbeta_data = dbeta.data_ptr<weight_t>();
    }
    GammaBetaBackwardDPCPPKernel<scalar_t, mean_t, weight_t>(
        N,
        C,
        group,
        mean_data,
        rstd_data,
        ds_data,
        db_data,
        dgamma_data,
        dbeta_data);
  }
}

void GroupNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
        if (gamma.defined() && gamma.scalar_type() == kFloat) {
          mean = at::empty({N, group}, X.options().dtype(kFloat));
          rstd = at::empty({N, group}, X.options().dtype(kFloat));
          GroupNormKernelImplInternal<scalar_t, float, float>(
              X,
              gamma,
              beta,
              N,
              C,
              HxW,
              group,
              static_cast<scalar_t>(eps),
              Y,
              mean,
              rstd);
        } else {
          mean = at::empty({N, group}, X.options());
          rstd = at::empty({N, group}, X.options());
          GroupNormKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              X,
              gamma,
              beta,
              N,
              C,
              HxW,
              group,
              static_cast<scalar_t>(eps),
              Y,
              mean,
              rstd);
        }
      });
}

void GroupNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormBackwardKernelImpl",
      [&]() {
        if (gamma.defined() && gamma.scalar_type() == kFloat) {
          GroupNormBackwardKernelImplInternal<scalar_t, float, float>(
              dY,
              X,
              mean,
              rstd,
              gamma,
              N,
              C,
              HxW,
              group,
              dX,
              dgamma,
              dbeta,
              grad_input_mask);
        } else {
          GroupNormBackwardKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              dY,
              X,
              mean,
              rstd,
              gamma,
              N,
              C,
              HxW,
              group,
              dX,
              dgamma,
              dbeta,
              grad_input_mask);
        }
      });
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt,
    const c10::optional<at::Tensor>& beta_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;

  c10::MaybeOwned<Tensor> beta_maybe_owned =
      at::borrow_from_optional_tensor(beta_opt);
  const Tensor& beta = *beta_maybe_owned;

  if (gamma.defined()) {
    TORCH_CHECK(
        gamma.scalar_type() == kFloat || X.scalar_type() == gamma.scalar_type(),
        "Input and gamma should be of the same datatype.");
  }
  if (beta.defined()) {
    TORCH_CHECK(
        beta.scalar_type() == kFloat || X.scalar_type() == beta.scalar_type(),
        "Input and beta should be of the same datatype.");
  }

  Tensor Y, mean, rstd;
  GroupNormKernelImpl(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<at::Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  if (gamma.defined()) {
    TORCH_CHECK(
        gamma.scalar_type() == kFloat || X.scalar_type() == gamma.scalar_type(),
        "Input and weight should be of the same datatype.");
  }

  Tensor dX, dgamma, dbeta;
  GroupNormBackwardKernelImpl(
      dY,
      X,
      mean,
      rstd,
      gamma,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta,
      grad_input_mask);
  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace AtenIpexTypeXPU
} // namespace at
