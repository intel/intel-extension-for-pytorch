#include "Norm.h"

using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int vec_size,
    int SIMD>
void RowwiseMomentsDPCPPKernel(
    int64_t Batch,
    int64_t Plane,
    int64_t WGPlane,
    int workgroup_num,
    int workgroup_num_foreach,
    int local_size,
    int sub_group_num,
    scalar_t eps,
    const Tensor& X,
    scalar_t* mean_data,
    scalar_t* rstd_data) {
  // X: [N * Group][C/Group * HxW]
  // workitem: [workgroup_num][workgroup_num_foreach][local_size]
  scalar_t* X_data = X.data_ptr<scalar_t>();

  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  sycl::range<3> local_range{1, 1, local_size};
  sycl::range<3> global_range{workgroup_num, workgroup_num_foreach, local_size};
  IndexType loops_end = WGPlane / vec_size;

  int sub_group_num_global = 1;
  Tensor semaphores, scratchpad;
  int* semaphores_ptr;
  accscalar_t* scratchpad_ptr;
  if (workgroup_num_foreach) {
    int scratchpad_size = 2 * Batch * workgroup_num_foreach;
    init_scratchpad<accscalar_t, SIMD>(
        X,
        semaphores,
        scratchpad,
        sub_group_num_global,
        workgroup_num,
        scratchpad_size,
        workgroup_num_foreach);
    semaphores_ptr = semaphores.data_ptr<int>();
    scratchpad_ptr = scratchpad.data_ptr<accscalar_t>();
  }

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    dpcpp_local_acc_t<bool> last_workgroup(1, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          IndexType local_id = item_id.get_local_id(2);
          IndexType group_id = item_id.get_group(0);
          IndexType group_id_foreach = item_id.get_group(1);
          IndexType group_offset = group_id * Plane;

          accscalar_t sum1 = accscalar_t(0);
          accscalar_t sum2 = accscalar_t(0);
          for (IndexType j = local_id; j < loops_end; j += local_size) {
            IndexType plane_offset = group_id_foreach * WGPlane + j * vec_size;
            if (plane_offset < Plane) {
              vec_t value = *(reinterpret_cast<vec_t*>(
                  X_data + group_offset + plane_offset));
#pragma unroll(vec_size)
              for (int v = 0; v < vec_size; ++v) {
                sum1 += static_cast<accscalar_t>(value[v]);
                sum2 += static_cast<accscalar_t>(value[v]) *
                    static_cast<accscalar_t>(value[v]);
              }
            }
          }

          norm_group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum1,
              sum2,
              local_sum1,
              local_sum2,
              [](accscalar_t a, accscalar_t b) { return a + b; });

          if (workgroup_num_foreach > 1) {
            norm_global_reduce<SIMD, accscalar_t, IndexType>(
                item_id,
                workgroup_num_foreach,
                local_size,
                sub_group_num_global,
                sum1,
                sum2,
                scratchpad_ptr,
                semaphores_ptr,
                local_sum1,
                local_sum2,
                last_workgroup,
                [](accscalar_t a, accscalar_t b) { return a + b; });

            if (last_workgroup[0] && local_id == 0) {
              project_and_store<scalar_t, accscalar_t>(
                  group_id, sum1, sum2, Plane, mean_data, rstd_data, eps);
            }
          } else {
            if (local_id == 0) {
              project_and_store<scalar_t, accscalar_t>(
                  group_id, sum1, sum2, Plane, mean_data, rstd_data, eps);
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t, int SIMD>
void RowwiseMomentsDPCPPKernelImpl(
    int64_t Batch,
    int64_t Plane,
    int vec_size,
    scalar_t eps,
    const Tensor& X,
    scalar_t* mean_data,
    scalar_t* rstd_data) {
  int workgroup_num, workgroup_num_foreach, local_size, sub_group_num;
  get_workgroup_size<SIMD>(
      Batch,
      Plane,
      vec_size,
      workgroup_num,
      workgroup_num_foreach,
      local_size,
      sub_group_num);
  int WGPlane = (Plane + workgroup_num_foreach - 1) / workgroup_num_foreach;

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index = canUse32BitIndexMath(X);

#define VecRowwiseMomentsDPCPPKernel(vec_size)                                 \
  {                                                                            \
    using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>; \
    constexpr int align_bytes = alignof(vec_t);                                \
    int X_start = ((uint64_t)X.data_ptr()) % align_bytes / sizeof(scalar_t);   \
    if (X_start == 0 && WGPlane % vec_size == 0) {                             \
      if (can_use_32bit_index) {                                               \
        RowwiseMomentsDPCPPKernel<                                             \
            scalar_t,                                                          \
            accscalar_t,                                                       \
            uint32_t,                                                          \
            vec_size,                                                          \
            SIMD>(                                                             \
            Batch,                                                             \
            Plane,                                                             \
            WGPlane,                                                           \
            workgroup_num,                                                     \
            workgroup_num_foreach,                                             \
            local_size,                                                        \
            sub_group_num,                                                     \
            eps,                                                               \
            X,                                                                 \
            mean_data,                                                         \
            rstd_data);                                                        \
      } else {                                                                 \
        RowwiseMomentsDPCPPKernel<                                             \
            scalar_t,                                                          \
            accscalar_t,                                                       \
            uint64_t,                                                          \
            vec_size,                                                          \
            SIMD>(                                                             \
            Batch,                                                             \
            Plane,                                                             \
            WGPlane,                                                           \
            workgroup_num,                                                     \
            workgroup_num_foreach,                                             \
            local_size,                                                        \
            sub_group_num,                                                     \
            eps,                                                               \
            X,                                                                 \
            mean_data,                                                         \
            rstd_data);                                                        \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
  }

  switch (vec_size) {
    case 8: {
      VecRowwiseMomentsDPCPPKernel(8);
    }
    case 4: {
      VecRowwiseMomentsDPCPPKernel(4);
    }
    case 2: {
      VecRowwiseMomentsDPCPPKernel(2);
    }
    default: {
      VecRowwiseMomentsDPCPPKernel(1);
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename weight_t>
void ComputeFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const scalar_t* mean_data,
    const scalar_t* rstd_data,
    const weight_t* gamma_data,
    const weight_t* beta_data,
    accscalar_t* a_data,
    accscalar_t* b_data) {
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

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int vec_size>
void GroupNormForwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const Tensor& X,
    const accscalar_t* a_data,
    const accscalar_t* b_data,
    Tensor& Y) {
  // input: [NxC][HxW]
  // a,b: [NxC]
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();

  IndexType Plane = C * HxW;
  IndexType loops_end = (N * C * HxW + vec_size - 1) / vec_size;
  bool channels_last = is_smf_channels_last(X);

  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int total_threads = dpcppMaxWorkItemsPerTile(dev_id);
  auto local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_threads), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item_id) {
          IndexType local_id = item_id.get_global_linear_id();
          for (IndexType i = local_id; i < loops_end; i += total_threads) {
            IndexType remaining = N * C * HxW - i * vec_size;
            if (remaining < vec_size) {
              for (int j = 0; j < remaining; ++j) {
                IndexType offset = i * vec_size + j;

                IndexType nc;
                if (channels_last) {
                  nc = offset / Plane * C + offset % C;
                } else {
                  nc = offset / HxW;
                }
                Y_data[offset] = static_cast<scalar_t>(
                    a_data[nc] * static_cast<accscalar_t>(X_data[offset]) +
                    b_data[nc]);
              }
            } else {
              IndexType offset = i * vec_size;

              vec_t in_val = *(reinterpret_cast<vec_t*>(X_data + offset));
              vec_t out_val;
#pragma unroll(vec_size)
              for (int v = 0; v < vec_size; ++v) {
                IndexType nc;
                if (channels_last) {
                  nc = (offset + v) / Plane * C + (offset + v) % C;
                } else {
                  nc = (offset + v) / HxW;
                }
                out_val[v] = static_cast<scalar_t>(
                    a_data[nc] * static_cast<accscalar_t>(in_val[v]) +
                    b_data[nc]);
              }
              *(reinterpret_cast<vec_t*>(Y_data + offset)) = out_val;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void GroupNormForwardDPCPPKernelImpl(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int vec_size,
    const Tensor& X,
    const accscalar_t* a_data,
    const accscalar_t* b_data,
    Tensor& Y) {
  auto X_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(X.data_ptr()));
  auto Y_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(Y.data_ptr()));
  vec_size = std::min(vec_size, X_vec_size);
  vec_size = std::min(vec_size, Y_vec_size);

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index = canUse32BitIndexMath(X);

#define VecGroupNormForwardDPCPPKernel(vec_size)                              \
  {                                                                           \
    if (can_use_32bit_index) {                                                \
      GroupNormForwardDPCPPKernel<scalar_t, accscalar_t, uint32_t, vec_size>( \
          N, C, HxW, X, a_data, b_data, Y);                                   \
    } else {                                                                  \
      GroupNormForwardDPCPPKernel<scalar_t, accscalar_t, uint64_t, vec_size>( \
          N, C, HxW, X, a_data, b_data, Y);                                   \
    }                                                                         \
    break;                                                                    \
  }

  switch (vec_size) {
    case 8: {
      VecGroupNormForwardDPCPPKernel(8);
    }
    case 4: {
      VecGroupNormForwardDPCPPKernel(4);
    }
    case 2: {
      VecGroupNormForwardDPCPPKernel(2);
    }
    case 1: {
      VecGroupNormForwardDPCPPKernel(1);
    }
  }
}

template <typename scalar_t, typename weight_t>
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
  const int64_t G = group;
  const int64_t D = C / G;
  int vec_size = get_vec_size<scalar_t>(N * C * HxW);
  Tensor X_cont = X.contiguous();

  mean = at::empty({N, group}, X.options());
  rstd = at::empty({N, group}, X.options());
  scalar_t* mean_data = mean.data_ptr<scalar_t>();
  scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
  constexpr int SIMD16 = 16;
  RowwiseMomentsDPCPPKernelImpl<scalar_t, accscalar_t, SIMD16>(
      N * G, D * HxW, vec_size, eps, X_cont, mean_data, rstd_data);

  const weight_t* gamma_data =
      gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  const weight_t* beta_data =
      beta.defined() ? beta.data_ptr<weight_t>() : nullptr;
  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  accscalar_t* a_data = a.data_ptr<accscalar_t>();
  accscalar_t* b_data = b.data_ptr<accscalar_t>();
  ComputeFusedParamsDPCPPKernel<scalar_t, accscalar_t, weight_t>(
      N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);

  // propagate channels_last format from X to Y
  if (is_smf_channels_last(X)) {
    X_cont = X;
    auto smf = X.suggest_memory_format();
    Y = at::empty_like(X, smf);
  } else {
    Y = at::empty_like(X_cont);
  }
  GroupNormForwardDPCPPKernelImpl<scalar_t, accscalar_t>(
      N, C, HxW, vec_size, X_cont, a_data, b_data, Y);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int vec_size,
    int SIMD>
void ComputeInternalGradientsDPCPPKernel(
    int64_t Batch,
    int64_t Plane,
    const Tensor& dY,
    const Tensor& X,
    accscalar_t* ds_data,
    accscalar_t* db_data,
    int workgroup_num,
    int workgroup_num_foreach,
    int local_size,
    int sub_group_num) {
  scalar_t* dY_data = dY.data_ptr<scalar_t>();
  scalar_t* X_data = X.data_ptr<scalar_t>();

  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  sycl::range<3> local_range{1, 1, local_size};
  sycl::range<3> global_range{workgroup_num, workgroup_num_foreach, local_size};
  auto WGPlane = Plane / workgroup_num_foreach;
  IndexType loops_end = WGPlane / vec_size;

  int sub_group_num_foreach = 1;
  Tensor semaphores, scratchpad;
  int* semaphores_ptr;
  accscalar_t* scratchpad_ptr;
  if (workgroup_num_foreach) {
    int scratchpad_size = 2 * Batch * workgroup_num_foreach;
    init_scratchpad<accscalar_t, SIMD>(
        X,
        semaphores,
        scratchpad,
        sub_group_num_foreach,
        workgroup_num,
        scratchpad_size,
        workgroup_num_foreach);
    semaphores_ptr = semaphores.data_ptr<int>();
    scratchpad_ptr = scratchpad.data_ptr<accscalar_t>();
  }

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    dpcpp_local_acc_t<bool> last_workgroup(1, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_range), sycl::range<3>(local_range)),
        [=](sycl::nd_item<3> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          auto local_id = item_id.get_local_id(2);
          auto group_id = item_id.get_group(0);
          IndexType group_id_foreach = item_id.get_group(1);
          IndexType group_offset = group_id * Plane;

          accscalar_t sum1 = 0;
          accscalar_t sum2 = 0;
          for (int64_t j = local_id; j < loops_end; j += local_size) {
            IndexType plane_offset = group_id_foreach * WGPlane + j * vec_size;
            if (plane_offset < Plane) {
              vec_t dY_val = *(reinterpret_cast<vec_t*>(
                  dY_data + group_offset + plane_offset));
              vec_t X_val = *(reinterpret_cast<vec_t*>(
                  X_data + group_offset + plane_offset));
#pragma unroll(vec_size)
              for (int v = 0; v < vec_size; ++v) {
                sum1 += static_cast<accscalar_t>(dY_val[v]) *
                    static_cast<accscalar_t>(X_val[v]);
                sum2 += static_cast<accscalar_t>(dY_val[v]);
              }
            }
          }

          norm_group_reduce<SIMD, accscalar_t>(
              item_id,
              sub_group_num,
              sum1,
              sum2,
              local_sum1,
              local_sum2,
              [](accscalar_t a, accscalar_t b) { return a + b; });

          if (workgroup_num_foreach > 1) {
            norm_global_reduce<SIMD, accscalar_t, IndexType>(
                item_id,
                workgroup_num_foreach,
                local_size,
                sub_group_num,
                sum1,
                sum2,
                scratchpad_ptr,
                semaphores_ptr,
                local_sum1,
                local_sum2,
                last_workgroup,
                [](accscalar_t a, accscalar_t b) { return a + b; });

            if (last_workgroup[0] && local_id == 0) {
              ds_data[group_id] = sum1;
              db_data[group_id] = sum2;
            }
          } else {
            if (local_id == 0) {
              ds_data[group_id] = sum1;
              db_data[group_id] = sum2;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t, int SIMD>
void ComputeInternalGradientsDPCPPKernelImpl(
    int64_t total_size,
    int64_t HxW,
    int vec_size,
    const Tensor& dY,
    const Tensor& X,
    accscalar_t* ds_data,
    accscalar_t* db_data) {
  int Batch = total_size;
  int Plane = HxW;

  int workgroup_num, workgroup_num_foreach, local_size, sub_group_num;
  get_workgroup_size<SIMD>(
      Batch,
      Plane,
      vec_size,
      workgroup_num,
      workgroup_num_foreach,
      local_size,
      sub_group_num);
  int WGPlane = (Plane + workgroup_num_foreach - 1) / workgroup_num_foreach;

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index =
      canUse32BitIndexMath(X) && canUse32BitIndexMath(dY);
#define VecComputeInternalGradientsDPCPPKernel(vec_size)                       \
  {                                                                            \
    using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>; \
    constexpr int align_bytes = alignof(vec_t);                                \
    int X_start = ((uint64_t)X.data_ptr()) % align_bytes / sizeof(scalar_t);   \
    int dY_start = ((uint64_t)dY.data_ptr()) % align_bytes / sizeof(scalar_t); \
    if (X_start == 0 && dY_start == 0 && WGPlane % vec_size == 0) {            \
      if (can_use_32bit_index) {                                               \
        ComputeInternalGradientsDPCPPKernel<                                   \
            scalar_t,                                                          \
            accscalar_t,                                                       \
            uint32_t,                                                          \
            vec_size,                                                          \
            SIMD>(                                                             \
            Batch,                                                             \
            Plane,                                                             \
            dY,                                                                \
            X,                                                                 \
            ds_data,                                                           \
            db_data,                                                           \
            workgroup_num,                                                     \
            workgroup_num_foreach,                                             \
            local_size,                                                        \
            sub_group_num);                                                    \
      } else {                                                                 \
        ComputeInternalGradientsDPCPPKernel<                                   \
            scalar_t,                                                          \
            accscalar_t,                                                       \
            uint64_t,                                                          \
            vec_size,                                                          \
            SIMD>(                                                             \
            Batch,                                                             \
            Plane,                                                             \
            dY,                                                                \
            X,                                                                 \
            ds_data,                                                           \
            db_data,                                                           \
            workgroup_num,                                                     \
            workgroup_num_foreach,                                             \
            local_size,                                                        \
            sub_group_num);                                                    \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
  }

  switch (vec_size) {
    case 8: {
      VecComputeInternalGradientsDPCPPKernel(8);
    }
    case 4: {
      VecComputeInternalGradientsDPCPPKernel(4);
    }
    case 2: {
      VecComputeInternalGradientsDPCPPKernel(2);
    }
    default: {
      VecComputeInternalGradientsDPCPPKernel(1);
    }
  }
}

template <typename T, typename weight_t>
void ComputeGradOutputCoeffientDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* rstd,
    const weight_t* gamma,
    acc_type<T>* c1) {
  using accscalar_t = acc_type<T>;
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

template <typename T, typename accscalar_t, typename weight_t, int SIMD>
void ComputeBackwardFusedParamsDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* mean,
    const T* rstd,
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
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  local_size = std::min(local_size, D);
  int sub_group_num = (local_size + SIMD - 1) / SIMD;
  local_size = sub_group_num * SIMD;
  auto cgf = DPCPP_Q_CGF(cgh) [[intel::reqd_sub_group_size(SIMD)]] {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(N, group * local_size),
            sycl::range<2>(1, local_size)),
        [=](sycl::nd_item<2> item_id) {
          auto local_id = item_id.get_local_id(1);
          auto n = item_id.get_group(0);
          auto g = item_id.get_group(1);
          auto ng = n * G + g;
          accscalar_t sum1 = 0;
          accscalar_t sum2 = 0;
          for (int64_t i = local_id; i < D; i += local_size) {
            auto nc = ng * D + i;
            auto c = g * D + i;
            const accscalar_t gamma_v = gamma == nullptr
                ? accscalar_t(1)
                : static_cast<accscalar_t>(gamma[c]);
            sum1 += ds[nc] * gamma_v;
            sum2 += db[nc] * gamma_v;
          }

          norm_group_reduce<SIMD, accscalar_t>(
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

template <typename scalar_t, typename accscalar_t, typename weight_t>
void GammaBetaBackwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const scalar_t* mean,
    const scalar_t* rstd,
    const accscalar_t* ds,
    const accscalar_t* db,
    weight_t* dgamma,
    weight_t* dbeta) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto local_size = dpcppMaxWorkGroupSize(dev_id);
  auto total_threads = ((C + local_size - 1) / local_size) * local_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_threads), sycl::range<1>(local_size)),
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

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int vec_size>
void GroupNormBackwardDPCPPKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const Tensor& dY,
    const Tensor& X,
    const accscalar_t* c1_data,
    const accscalar_t* c2_data,
    const accscalar_t* c3_data,
    Tensor& dX) {
  int64_t D = C / group;
  int64_t Plane = C * HxW;
  int64_t Numel = N * Plane;
  scalar_t* dY_ptr = dY.data_ptr<scalar_t>();
  scalar_t* X_ptr = X.data_ptr<scalar_t>();
  scalar_t* dX_ptr = dX.data_ptr<scalar_t>();

  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int total_threads = dpcppMaxWorkItemsPerTile(dev_id);
  auto local_size = dpcppMaxWorkGroupSize(dev_id);

  IndexType loops_end = (Numel + vec_size - 1) / vec_size;
  bool channels_last = is_smf_channels_last(X);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_threads), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item_id) {
          IndexType local_id = item_id.get_global_linear_id();
          for (IndexType i = local_id; i < loops_end; i += total_threads) {
            IndexType remaining = Numel - i * vec_size;
            if (remaining < vec_size) {
              for (int j = 0; j < remaining; ++j) {
                IndexType offset = i * vec_size + j;

                IndexType nc, ng;
                if (channels_last) {
                  nc = offset / Plane * C + offset % C;
                  ng = nc / D;
                } else {
                  nc = offset / HxW;
                  ng = nc / D;
                }
                dX_ptr[offset] =
                    c1_data[nc] * static_cast<accscalar_t>(dY_ptr[offset]) +
                    c2_data[ng] * static_cast<accscalar_t>(X_ptr[offset]) +
                    c3_data[ng];
              }
            } else {
              IndexType offset = i * vec_size;

              vec_t X_val = *(reinterpret_cast<vec_t*>(X_ptr + offset));
              vec_t dY_val = *(reinterpret_cast<vec_t*>(dY_ptr + offset));
              vec_t dX_val;
#pragma unroll(vec_size)
              for (int v = 0; v < vec_size; ++v) {
                IndexType nc, ng;
                if (channels_last) {
                  nc = (offset + v) / Plane * C + (offset + v) % C;
                  ng = nc / D;
                } else {
                  nc = (offset + v) / HxW;
                  ng = nc / D;
                }
                dX_val[v] = c1_data[nc] * static_cast<accscalar_t>(dY_val[v]) +
                    c2_data[ng] * static_cast<accscalar_t>(X_val[v]) +
                    c3_data[ng];
              }
              *(reinterpret_cast<vec_t*>(dX_ptr + offset)) = dX_val;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void GroupNormBackwardDPCPPKernelImpl(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    int vec_size,
    const Tensor& dY,
    const Tensor& X,
    const accscalar_t* c1_data,
    const accscalar_t* c2_data,
    const accscalar_t* c3_data,
    Tensor& dX) {
  auto dY_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(dY.data_ptr()));
  auto X_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(X.data_ptr()));
  auto dX_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(dX.data_ptr()));
  vec_size = std::min(vec_size, dY_vec_size);
  vec_size = std::min(vec_size, X_vec_size);
  vec_size = std::min(vec_size, dX_vec_size);

  bool can_use_32bit_index =
      canUse32BitIndexMath(X) && canUse32BitIndexMath(dY);
#define VecGroupNormBackwardDPCPPKernel(vec_size)                              \
  {                                                                            \
    if (can_use_32bit_index) {                                                 \
      GroupNormBackwardDPCPPKernel<scalar_t, accscalar_t, uint32_t, vec_size>( \
          N, C, HxW, group, dY, X, c1_data, c2_data, c3_data, dX);             \
    } else {                                                                   \
      GroupNormBackwardDPCPPKernel<scalar_t, accscalar_t, uint64_t, vec_size>( \
          N, C, HxW, group, dY, X, c1_data, c2_data, c3_data, dX);             \
    }                                                                          \
    break;                                                                     \
  }

  switch (vec_size) {
    case 8: {
      VecGroupNormBackwardDPCPPKernel(8);
    }
    case 4: {
      VecGroupNormBackwardDPCPPKernel(4);
    }
    case 2: {
      VecGroupNormBackwardDPCPPKernel(2);
    }
    case 1: {
      VecGroupNormBackwardDPCPPKernel(1);
    }
  }
}

template <typename scalar_t, typename weight_t>
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
  const int64_t G = group;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  if (N == 0) {
    return;
  }

  constexpr int SIMD16 = 16;
  int vec_size = get_vec_size<scalar_t>(N * C * HxW);
  Tensor X_cont = X.contiguous();
  Tensor dY_cont = dY.contiguous();
  const scalar_t* mean_data = mean.data_ptr<scalar_t>();
  const scalar_t* rstd_data = rstd.data_ptr<scalar_t>();
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
  ComputeInternalGradientsDPCPPKernelImpl<scalar_t, accscalar_t, SIMD16>(
      N * C, HxW, vec_size, dY_cont, X_cont, ds_data, db_data);

  // compute gradient input (dX)
  if (grad_input_mask[0]) {
    Tensor c1 = at::empty({N, C}, X.options().dtype(kAccType));
    accscalar_t* c1_data = c1.data_ptr<accscalar_t>();
    ComputeGradOutputCoeffientDPCPPKernel<scalar_t, weight_t>(
        N, C, G, rstd_data, gamma_data, c1_data);

    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    accscalar_t* c2_data = c2.data_ptr<accscalar_t>();
    accscalar_t* c3_data = c3.data_ptr<accscalar_t>();
    ComputeBackwardFusedParamsDPCPPKernel<
        scalar_t,
        accscalar_t,
        weight_t,
        SIMD16>(
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

    if (is_smf_channels_last(X)) {
      auto smf = X.suggest_memory_format();
      dX = at::empty_like(X, smf);
      X_cont = X;
      dY_cont = dY.contiguous(smf);
    } else {
      dX = at::empty_like(X_cont);
    }
    GroupNormBackwardDPCPPKernelImpl<scalar_t>(
        N, C, HxW, G, vec_size, dY_cont, X_cont, c1_data, c2_data, c3_data, dX);
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
    GammaBetaBackwardDPCPPKernel<scalar_t, accscalar_t, weight_t>(
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
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
        if (gamma.scalar_type() == kFloat) {
          GroupNormKernelImplInternal<scalar_t, float>(
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
          GroupNormKernelImplInternal<scalar_t, scalar_t>(
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
        if (gamma.scalar_type() == kFloat) {
          GroupNormBackwardKernelImplInternal<scalar_t, float>(
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
          GroupNormBackwardKernelImplInternal<scalar_t, scalar_t>(
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
