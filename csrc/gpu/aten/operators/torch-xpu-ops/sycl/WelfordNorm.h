#pragma once

#include <ATen/native/Resize.h>
#include "aten/operators/torch-xpu-ops/comm/SYCLContext.h"
#include "aten/operators/torch-xpu-ops/comm/XPUMathCompat.h"
#include "aten/operators/torch-xpu-ops/sycl/MemoryAccess.h"

namespace at::native::xpu {

template <typename T>
inline T divup(T a, T b) {
  return (a + b - 1) / b;
}

std::tuple<int, int, int, int> get_adaptive_config(
    const int reduction,
    const int n_channels,
    const int vec_size,
    int max_wg_size,
    int loops_per_item = 8) {
  loops_per_item /= vec_size;
  int group_size_x = std::min(last_pow2(n_channels / vec_size), 32);
  int group_size_y = std::min(
      last_pow2(divup(reduction, loops_per_item)), max_wg_size / group_size_x);
  if (group_size_x * group_size_y != max_wg_size) {
    group_size_x =
        std::min(last_pow2(n_channels / vec_size), max_wg_size / group_size_y);
  }

  int nwg_x = divup(n_channels, group_size_x * vec_size);
  int nwg_y = std::min(
      divup(reduction, group_size_y * loops_per_item),
      int(syclMaxWorkItemsPerTile()) / (nwg_x * group_size_x) / (group_size_y));
  nwg_y = std::max(nwg_y, 1);

  // it's not worth having reduction between work groups if the reduction
  // dimension is not big enough
  nwg_y = nwg_y < 4 ? 1 : nwg_y;

  return std::make_tuple(group_size_y, group_size_x, nwg_y, nwg_x);
}

template <typename T, typename C>
inline void welford_merge(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0) / std::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

template <int VEC_SIZE, typename T, typename C, typename TACC, typename CACC>
inline void welford_vertical_merge(
    sycl::nd_item<2>& item,
    C& count,
    T& mean,
    T& m2n,
    CACC& shmem_count,
    TACC& shmem_mean,
    TACC& shmem_m2n) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();
#pragma unroll
  for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
    if (item.get_local_id(0) < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    item.barrier(sycl_local_fence);
    if (item.get_local_id(0) < offset &&
        item.get_local_id(0) + offset < item.get_local_range(0)) {
      auto address = address_base + offset * item.get_local_range(1);
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        welford_merge(
            count[v], mean[v], m2n[v], count_new[v], mean_new[v], m2n_new[v]);
      }
    }
  }
}

template <
    typename VarTransform,
    typename scalar_t,
    typename acc_t,
    int VEC_SIZE = 2>
struct WelfordBatchNormStatChannelsLastVecKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using vec_t = memory::aligned_vector<scalar_t, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<acc_t, VEC_SIZE>;
  using int_vec_t = memory::aligned_vector<int, VEC_SIZE>;

  void operator()(sycl::nd_item<2> item) const {
    //  init private counter
    acc_vec_t mean;
    acc_vec_t m2n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      mean[v] = acc_t(0);
      m2n[v] = acc_t(0);
      count[v] = int(0);
    }

    int gy = item.get_group(0);
    int gx = item.get_group(1);
    int c_vec_offset = item.get_global_id(1) * VEC_SIZE;
    int num_cooperative_groups = item.get_group_range(0);
    int inner_loop_stride = item.get_local_range(0) * num_cooperative_groups;

    for (int m_offset = item.get_global_id(0); m_offset < reduction_size_;
         m_offset += inner_loop_stride) {
      if (c_vec_offset < n_channels_) {
        int address_vec_base = m_offset * n_channels_ + c_vec_offset;
        auto input_vec = *reinterpret_cast<vec_t*>(
            const_cast<scalar_t*>(&input_[address_vec_base]));
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          auto x = input_vec[v];
          count[v]++;
          acc_t delta0 = x - mean[v];
          mean[v] += delta0 / count[v];
          acc_t delta1 = x - mean[v];
          m2n[v] += delta0 * delta1;
        }
      }
    }

    welford_vertical_merge<VEC_SIZE>(
        item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);

    // welford vertical merge
    if (num_cooperative_groups > 1) {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n = &staging_data_[n_channels_ * num_cooperative_groups];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[n_channels_ * num_cooperative_groups]);
      int address_vec_base = c_vec_offset + gy * n_channels_;

      // write data to staging_data;
      if (item.get_local_id(0) == 0) {
        *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]) = mean;
        *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]) = m2n;
        *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]) = count;
      }
      item.barrier(sycl_local_fence);

      // mark group done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[gx]);
        int old = atomic_count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_group_done_[0] = (old == (num_cooperative_groups - 1));
      }
      item.barrier(sycl_local_fence);

      // check that all data is now available in global memory
      if (is_last_group_done_[0]) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          mean[v] = acc_t(0);
          m2n[v] = acc_t(0);
          count[v] = int(0);
        }

        for (int y = item.get_local_id(0); y < num_cooperative_groups;
             y += item.get_local_range(0)) {
          if (c_vec_offset < n_channels_) {
            address_vec_base = y * n_channels_ + c_vec_offset;
            auto mean_new =
                *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]);
            auto m2n_new =
                *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]);
            auto count_new =
                *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]);
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
              welford_merge(
                  count[v],
                  mean[v],
                  m2n[v],
                  count_new[v],
                  mean_new[v],
                  m2n_new[v]);
            }
          }
        }
        welford_vertical_merge<VEC_SIZE>(
            item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
      }
    }

    if (item.get_local_id(0) == 0 &&
        (num_cooperative_groups == 1 || is_last_group_done_[0]) &&
        c_vec_offset < n_channels_) {
      acc_vec_t invstd_vec;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
      }

      *reinterpret_cast<acc_vec_t*>(&save_mean_[c_vec_offset]) = mean;
      *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_vec_offset]) = invstd_vec;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    auto local_size = group_size_x_ * group_size_y_;
    shmem_mean_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_m2n_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_count_ = sycl_local_acc_t<int_vec_t>(sycl::range<1>(local_size), cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>(1), cgh);
  }

  WelfordBatchNormStatChannelsLastVecKernelFunctor(
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd,
      int reduction_size,
      int n_channels,
      acc_t* staging_data,
      int* semaphores,
      double epsilon)
      : input_(input),
        save_mean_(save_mean),
        save_invstd_(save_invstd),
        reduction_size_(reduction_size),
        n_channels_(n_channels),
        staging_data_(staging_data),
        semaphores_(semaphores),
        epsilon_(epsilon) {}

  void init() {
    using KernelT = WelfordBatchNormStatChannelsLastVecKernelFunctor<
        VarTransform,
        scalar_t,
        acc_t,
        VEC_SIZE>;
    auto max_group_size = syclMaxWorkGroupSize<KernelT>();
    std::tie(group_size_y_, group_size_x_, ngroups_y_, ngroups_x_) =
        get_adaptive_config(
            reduction_size_, n_channels_, VEC_SIZE, max_group_size);
  }

  static bool valid(
      int reduction_size,
      int n_channels,
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd) {
    bool valid = sizeof(scalar_t) <= 2;
    valid = valid && (n_channels % VEC_SIZE == 0);
    valid = valid &&
        (memory::can_vectorize_up_to<scalar_t>((char*)input) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_mean) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_invstd) >= VEC_SIZE);
    return valid;
  }

  sycl::range<2> local_range() const {
    return sycl::range<2>(group_size_y_, group_size_x_);
  }

  sycl::range<2> global_range() const {
    return sycl::range<2>(
        group_size_y_ * ngroups_y_, group_size_x_ * ngroups_x_);
  }

  int staging_size() const {
    return ngroups_y_ * n_channels_ * 4;
  }

  int semaphores_size() const {
    return ngroups_x_;
  }

  bool set_staging_data_check(acc_t* staging_data) {
    staging_data_ = staging_data;
    return (
        (staging_data == nullptr) ||
        (memory::can_vectorize_up_to<acc_t>((char*)staging_data) >= VEC_SIZE));
  }

  void set_semaphores(int* semaphores) {
    semaphores_ = semaphores;
  }

  int num_cooperative_groups() const {
    return ngroups_y_;
  }

 private:
  const scalar_t* input_;
  acc_t* save_mean_;
  acc_t* save_invstd_;
  int reduction_size_;
  int n_channels_;
  acc_t* staging_data_;
  int* semaphores_;
  double epsilon_;

  size_t group_size_y_;
  size_t group_size_x_;
  size_t ngroups_y_;
  size_t ngroups_x_;

  sycl_local_acc_t<acc_vec_t> shmem_mean_;
  sycl_local_acc_t<acc_vec_t> shmem_m2n_;
  sycl_local_acc_t<int_vec_t> shmem_count_;
  sycl_local_acc_t<bool> is_last_group_done_;
};

} // namespace at::native::xpu
