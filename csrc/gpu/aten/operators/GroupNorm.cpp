#include <ATen/ATen.h>

#include <core/MemoryFormat.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
static void RowwiseMomentsDPCPPKernel(
    int64_t total_size,
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  auto global_size = total_size * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item_id) {
          auto local_id = item_id.get_local_id(0);
          auto i = item_id.get_group(0);
          auto g = item_id.get_group();

          T_ACC sum1 = 0;
          T_ACC sum2 = 0;
          for (int64_t j = local_id; j < N; j += local_size) {
            const int64_t index = i * N + j;
            sum1 += static_cast<T_ACC>(X[index]);
            sum2 += static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(X[index]);
          }
          sum1 = sycl::reduce_over_group(g, sum1, sycl::ext::oneapi::plus<>());
          sum2 = sycl::reduce_over_group(g, sum2, sycl::ext::oneapi::plus<>());
          if (local_id == 0) {
            const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(N);
            sum1 *= scale;
            sum2 = sum2 * scale - sum1 * sum1;
            if (sum2 < 0) {
              sum2 = T_ACC(0);
            }
            mean[i] = sum1;
            rstd[i] = Numerics<T_ACC>::rsqrt(sum2 + static_cast<T_ACC>(eps));
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
static void ComputeFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    acc_type<T>* a,
    acc_type<T>* b) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = N * C;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
      auto id = itemId.get_id(0);

      const int64_t ng = id / (C / group);
      const int64_t c = id % C;
      const T_ACC x = (gamma == nullptr)
          ? static_cast<T_ACC>(rstd[ng])
          : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
      a[id] = x;
      b[id] = -x * static_cast<T_ACC>(mean[ng]) +
          (beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[c]));
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
void GroupNormForwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* X,
    const acc_type<T>* a,
    const acc_type<T>* b,
    T* Y) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  if (HxW < local_size) {
    auto total_threads =
        ((N * C * HxW + local_size - 1) / local_size) * local_size;
    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(total_threads), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> item_id) {
            auto local_id = item_id.get_local_id(0);
            auto group_id = item_id.get_group(0);
            const int64_t index = group_id * local_size + local_id;
            if (index < N * C * HxW) {
              const int64_t nc = index / HxW;
              Y[index] = a[nc] * static_cast<T_ACC>(X[index]) + b[nc];
            }
          });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  } else {
    auto total_threads = N * C * local_size;
    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(total_threads), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> item_id) {
            auto local_id = item_id.get_local_id(0);
            auto group_id = item_id.get_group(0);
            const int64_t nc = group_id;
            for (int64_t hw = local_id; hw < HxW; hw += local_size) {
              const int64_t index = nc * HxW + hw;
              Y[index] = a[nc] * static_cast<T_ACC>(X[index]) + b[nc];
            }
          });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
}

template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  using T_ACC = acc_type<T>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* a_data = a.data_ptr<T_ACC>();
  T_ACC* b_data = b.data_ptr<T_ACC>();

  RowwiseMomentsDPCPPKernel<T>(
      N * G, D * HxW, eps, X_data, mean_data, rstd_data);

  ComputeFusedParamsDPCPPKernel<T>(
      N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);

  GroupNormForwardDPCPPKernel<T>(N, C, HxW, X_data, a_data, b_data, Y_data);
}

template <typename T>
void ComputeInternalGradientsDPCPPKernel(
    int64_t total_size,
    int64_t HxW,
    const T* dY,
    const T* X,
    acc_type<T>* ds,
    acc_type<T>* db) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  auto global_size = total_size * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item_id) {
          auto local_id = item_id.get_local_id(0);
          auto nc = item_id.get_group(0);
          auto g = item_id.get_group();
          T_ACC sum1 = 0;
          T_ACC sum2 = 0;
          for (int64_t hw = local_id; hw < HxW; hw += local_size) {
            const int64_t index = nc * HxW + hw;
            sum1 +=
                static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]);
            sum2 += static_cast<T_ACC>(dY[index]);
          }
          sum1 = sycl::reduce_over_group(g, sum1, sycl::ext::oneapi::plus<>());
          sum2 = sycl::reduce_over_group(g, sum2, sycl::ext::oneapi::plus<>());
          if (local_id == 0) {
            ds[nc] = sum1;
            db[nc] = sum2;
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
void ComputeGradOutputCoeffientDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* rstd,
    const T* gamma,
    acc_type<T>* c1) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = N * C;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
      auto nc = itemId.get_id(0);

      const int64_t ng = nc / (C / group);
      const int64_t c = nc % C;
      c1[nc] = static_cast<T_ACC>(rstd[ng]) *
          (gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]));
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
void ComputeBackwardFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const acc_type<T>* ds,
    const acc_type<T>* db,
    acc_type<T>* c2,
    acc_type<T>* c3) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  auto total_threads = N * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(total_threads, group),
            sycl::range<2>(local_size, 1)),
        [=](sycl::nd_item<2> itemId) {
          using T_ACC = acc_type<T>;
          auto G = group;
          auto D = C / G;
          auto local_id = itemId.get_local_id(0);
          auto n = itemId.get_group(0);
          auto g = itemId.get_group(1);
          auto group_id = itemId.get_group();
          auto ng = n * G + g;
          T_ACC sum1 = 0;
          T_ACC sum2 = 0;
          for (int64_t i = local_id; i < D; i += local_size) {
            auto index = ng * D + i;
            auto c = g * D + i;
            const T_ACC gamma_v =
                gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
            sum1 += ds[index] * gamma_v;
            sum2 += db[index] * gamma_v;
          }
          sum1 = sycl::reduce_over_group(
              group_id, sum1, sycl::ext::oneapi::plus<>());
          sum2 = sycl::reduce_over_group(
              group_id, sum2, sycl::ext::oneapi::plus<>());
          if (local_id == 0) {
            const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
            const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
                static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
                static_cast<T_ACC>(rstd[ng]) * s;
            c2[ng] = x;
            c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
                sum2 * static_cast<T_ACC>(rstd[ng]) * s;
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
void GroupNormBackwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const acc_type<T>* c1,
    const acc_type<T>* c2,
    const acc_type<T>* c3,
    T* dX) {
  using T_ACC = acc_type<T>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  if (HxW < local_size) {
    auto total_threads =
        ((N * C * HxW + local_size - 1) / local_size) * local_size;
    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(total_threads), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> itemId) {
            auto index = itemId.get_global_linear_id();
            if (index < N * C * HxW) {
              auto nc = index / HxW;
              auto ng = nc / (C / group);
              dX[index] = c1[nc] * static_cast<T_ACC>(dY[index]) +
                  c2[ng] * static_cast<T_ACC>(X[index]) + c3[ng];
            }
          });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  } else {
    auto total_threads = N * C * local_size;
    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(total_threads), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> itemId) {
            auto local_id = itemId.get_local_id(0);
            auto group_id = itemId.get_group(0);
            auto D = C / group;
            auto ng = group_id / D;
            for (int64_t hw = local_id; hw < HxW; hw += local_size) {
              auto index = group_id * HxW + hw;
              dX[index] = c1[group_id] * static_cast<T_ACC>(dY[index]) +
                  c2[ng] * static_cast<T_ACC>(X[index]) + c3[ng];
            }
          });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
}

template <typename T>
void GammaBetaBackwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const acc_type<T>* ds,
    const acc_type<T>* db,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T>;
  // if (N < 512) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  auto total_threads = ((C + local_size - 1) / local_size) * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_threads), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> itemId) {
          auto index = itemId.get_global_linear_id();
          if (index < C) {
            auto G = group;
            auto D = C / G;
            T_ACC sum1 = 0;
            T_ACC sum2 = 0;
            for (int64_t n = 0; n < N; ++n) {
              auto nc = n * C + index;
              auto ng = n * G + index / D;
              sum1 += (dgamma == nullptr)
                  ? T_ACC(0)
                  : ((ds[nc] - db[nc] * static_cast<T_ACC>(mean[ng])) *
                     static_cast<T_ACC>(rstd[ng]));
              sum2 += (dbeta == nullptr) ? T_ACC(0) : db[nc];
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
  // Optimazed kernel for N size larger than 512
  /*} else {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
    const int64_t B = ((C + kReduceTileSize - 1) / kReduceTileSize) *
  kReduceTileSize; constexpr int kThreadX = kReduceTileSize; constexpr int
  kThreadY = kReduceTileSize / 2; cgh.parallel_for( sycl::nd_range<2>(
          sycl::range<2>(B, kThreadY), sycl::range<2>(kThreadX, kThreadY)),
      [=](sycl::nd_item<2> itemId) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    T_ACC dg_sum1 = 0;
    T_ACC dg_sum2 = 0;
    T_ACC db_sum1 = 0;
    T_ACC db_sum2 = 0;
    if (c < C) {
      const int64_t G = group;
      const int64_t D = C / G;
      for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
        const int64_t n1 = n;
        const int64_t n2 = n + blockDim.y;
        const int64_t nc1 = n1 * C + c;
        const int64_t nc2 = n2 * C + c;
        const int64_t ng1 = n1 * G + c / D;
        const int64_t ng2 = n2 * G + c / D;
        dg_sum1 += dgamma == nullptr
            ? T_ACC(0)
            : ((ds[nc1] - db[nc1] * static_cast<T_ACC>(mean[ng1])) *
              static_cast<T_ACC>(rstd[ng1]));
        db_sum1 += dbeta == nullptr ? T_ACC(0) : db[nc1];
        if (n2 < N) {
          dg_sum2 += dgamma == nullptr
              ? T_ACC(0)
              : ((ds[nc2] - db[nc2] * static_cast<T_ACC>(mean[ng2])) *
                static_cast<T_ACC>(rstd[ng2]));
          db_sum2 += dbeta == nullptr ? T_ACC(0) : db[nc2];
        }
      }
    }
    sum1 = sycl::reduce_over_group(group_id, sum1,
  sycl::ext::oneapi::plus<>()) sum2 = sycl::reduce_over_group(group_id,
  sum2, sycl::ext::oneapi::plus<>()) if (threadIdx.x == 0) { const int64_t
  c = blockIdx.x * blockDim.x + threadIdx.y; if (c < C) { if (dgamma !=
  nullptr) { dgamma[c] = sum1;
        }
        if (dbeta != nullptr) {
          dbeta[c] = sum2;
        }
      }
    }
    sum1 = sycl::reduce_over_group(group_id, sum1,
  sycl::ext::oneapi::plus<>()) sum2 = sycl::reduce_over_group(group_id,
  sum2, sycl::ext::oneapi::plus<>()) if (threadIdx.x == 0) { const int64_t
  c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y; if (c < C) { if
  (dgamma != nullptr) { dgamma[c] = sum1;
        }
        if (dbeta != nullptr) {
          dbeta[c] = sum2;
        }
      }
    }
  }*/
}

template <typename T>
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
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  using T_ACC = acc_type<T>;
  const int64_t G = group;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  if (N == 0) {
    return;
  }

  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* mean_data = mean.data_ptr<T>();
  const T* rstd_data = rstd.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  ComputeInternalGradientsDPCPPKernel<T>(
      N * C, HxW, dY_data, X_data, ds_data, db_data);

  if (dX_data != nullptr) {
    Tensor c1 = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c1_data = c1.data_ptr<T_ACC>();
    T_ACC* c2_data = c2.data_ptr<T_ACC>();
    T_ACC* c3_data = c3.data_ptr<T_ACC>();

    ComputeGradOutputCoeffientDPCPPKernel<T>(
        N, C, G, rstd_data, gamma_data, c1_data);

    ComputeBackwardFusedParamsDPCPPKernel<T>(
        N,
        C,
        HxW,
        G,
        mean_data,
        rstd_data,
        gamma_data,
        ds_data,
        db_data,
        c2_data,
        c3_data);

    GroupNormBackwardDPCPPKernel<T>(
        N, C, HxW, G, dY_data, X_data, c1_data, c2_data, c3_data, dX_data);
  }
  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data = dgamma->defined() ? dgamma->data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->data_ptr<T>() : nullptr;
    GammaBetaBackwardDPCPPKernel<T>(
        N,
        C,
        G,
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
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
        GroupNormKernelImplInternal<scalar_t>(
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
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormBackwardKernelImpl",
      [&]() {
        GroupNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
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
  auto smf = X.suggest_memory_format();
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;

  c10::MaybeOwned<Tensor> beta_maybe_owned =
      at::borrow_from_optional_tensor(beta_opt);
  const Tensor& beta = *beta_maybe_owned;

  Tensor X_cont = X.contiguous();
  Tensor Y = at::empty_like(X_cont);
  Tensor mean = at::empty({N, group}, X.options());
  Tensor rstd = at::empty({N, group}, X.options());
  GroupNormKernelImpl(
      X_cont, gamma, beta, N, C, HxW, group, eps, &Y, &mean, &rstd);
  if (is_smf_channels_last(X)) {
    Y = Y.contiguous(smf);
  }
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
  auto smf = X.suggest_memory_format();

  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  Tensor X_cont = X.contiguous();
  if (grad_input_mask[0]) {
    dX = at::empty_like(X_cont);
  }

  if (grad_input_mask[1]) {
    dgamma = at::empty_like(gamma);
  }
  if (grad_input_mask[2]) {
    dbeta = at::empty_like(gamma);
  }
  GroupNormBackwardKernelImpl(
      dY.contiguous(),
      X_cont,
      mean,
      rstd,
      gamma,
      N,
      C,
      HxW,
      group,
      &dX,
      &dgamma,
      &dbeta);
  if (is_smf_channels_last(X)) {
    dX = dX.contiguous(smf);
  }

  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace AtenIpexTypeXPU
} // namespace at
