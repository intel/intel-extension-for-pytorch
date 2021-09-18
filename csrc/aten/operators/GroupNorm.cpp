#include <ATen/ATen.h>

#include <runtime/Utils.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"

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
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
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
          sum1 =
              sycl::reduce_over_group(g, sum1, cl::sycl::ext::oneapi::plus<>());
          sum2 =
              sycl::reduce_over_group(g, sum2, cl::sycl::ext::oneapi::plus<>());
          if (local_id == 0) {
            const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(N);
            sum1 *= scale;
            sum2 = sum2 * scale - sum1 * sum1;
            if (sum2 < 0) {
              sum2 = T_ACC(0);
            }
            mean[i] = sum1;
            rstd[i] = DPCPP::rsqrt(sum2 + static_cast<T_ACC>(eps));
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
    cgh.parallel_for(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
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
          DPCPP::nd_range<1>(
              DPCPP::range<1>(total_threads), DPCPP::range<1>(local_size)),
          [=](DPCPP::nd_item<1> item_id) {
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
          DPCPP::nd_range<1>(
              DPCPP::range<1>(total_threads), DPCPP::range<1>(local_size)),
          [=](DPCPP::nd_item<1> item_id) {
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
  auto global_size = ((total_size + local_size - 1) / local_size) * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
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
          sum1 =
              sycl::reduce_over_group(g, sum1, cl::sycl::ext::oneapi::plus<>());
          sum2 =
              sycl::reduce_over_group(g, sum2, cl::sycl::ext::oneapi::plus<>());
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
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_threads), DPCPP::range<1>(group)),
        [=](DPCPP::nd_item<1> itemId) {
          auto id = itemId.get_global_linear_id();
          const int64_t ng = id / (C / group);
          const int64_t c = id % C;
          c1[id] = static_cast<T_ACC>(rstd[ng]) *
              (gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]));
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
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
  std::cout << "Groupnorm backward is not implemented yet";
  return;
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  Tensor Y = at::empty_like(X);
  Tensor mean = at::empty({N, group}, X.options());
  Tensor rstd = at::empty({N, group}, X.options());
  GroupNormKernelImpl(X, gamma, beta, N, C, HxW, group, eps, &Y, &mean, &rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::empty_like(X);
  }
  if (grad_input_mask[1]) {
    dgamma = at::empty_like(gamma);
  }
  if (grad_input_mask[2]) {
    dbeta = at::empty_like(gamma);
  }
  GroupNormBackwardKernelImpl(
      dY, X, mean, rstd, gamma, N, C, HxW, group, &dX, &dgamma, &dbeta);
  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace AtenIpexTypeXPU
} // namespace at
