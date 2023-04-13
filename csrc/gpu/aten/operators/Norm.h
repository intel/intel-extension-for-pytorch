#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace normalization {
template <typename scalar_t>
static int get_vec_size(size_t problem_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int total_resource = dpcppMaxWorkItemsPerTile(dev_id);

  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  int vec_size = max_vec_size;
  while ((vec_size >> 1) * total_resource > problem_size &&
         (vec_size >> 1) >= 1) {
    vec_size = vec_size >> 1;
  }
  return vec_size;
}

// get resource size for Reduce problem [Batch, Plane]
// the reduce is performed on Plane dimension
template <int SIMD>
static void get_workgroup_size(
    size_t Batch,
    size_t Plane,
    int vec_size,
    int& workgroup_num,
    int& workgroup_num_foreach,
    int& workgroup_size,
    int& sub_group_num) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  int total_resource = dpcppMaxWorkItemsPerTile(dev_id);
  workgroup_num = total_resource / max_workgroup_size;
  int max_workgroup_num_foreach = 1;
  workgroup_size = max_workgroup_size;

  // To keep high occupancy, we should activate at least workgroup_num number of
  // WG if Batch is larger than workgroup_num, use only one WG to process Plane
  // elements if Batch is smaller than workgroup_num, use workgroup_num_foreach
  // to process Plan elements
  while (workgroup_num > Batch) {
    workgroup_num = workgroup_num >> 1;
    max_workgroup_num_foreach = max_workgroup_num_foreach << 1;
  }
  workgroup_num_foreach =
      (Plane + workgroup_size * vec_size - 1) / (workgroup_size * vec_size);
  workgroup_num_foreach =
      std::min(workgroup_num_foreach, max_workgroup_num_foreach);
  // Reduce will waste the EU resource, then
  // minimize the workgroup_size and maximize the workgroup_num
  while (workgroup_num << 1 <= Batch && workgroup_size >= SIMD) {
    workgroup_num = workgroup_num << 1;
    workgroup_size = workgroup_size >> 1;
  }

  // Workgroup_num should larger or equal to Batch
  workgroup_num = std::max(workgroup_num, int(Batch));
  // At least one subgroup for reduce
  sub_group_num = (workgroup_size + SIMD - 1) / SIMD;
}

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void norm_group_reduce(
    nd_item_id item_id,
    int sub_group_num,
    accscalar_t& mean,
    accscalar_t& rstd,
    const local_shared& local_mean,
    const local_shared& local_rstd,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    mean = bin_op(mean, static_cast<accscalar_t>(sg.shuffle_down(mean, i)));
    rstd = bin_op(rstd, static_cast<accscalar_t>(sg.shuffle_down(rstd, i)));
  }
  if (sub_group_num == 1) {
    mean = sycl::group_broadcast(sg, mean, 0);
    rstd = sycl::group_broadcast(sg, rstd, 0);
    return;
  }

  uint32_t sg_local_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();
  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  int idx = sg_id;
  if (sg_local_id == 0) {
    local_mean[sg_id] = mean;
    local_rstd[sg_id] = rstd;
  }
  item_id.barrier(dpcpp_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (idx == 0) {
    mean = 0;
    rstd = 0;
    if (sg_local_id < sub_group_num) {
      mean = accscalar_t(local_mean[sg_local_id]);
      rstd = accscalar_t(local_rstd[sg_local_id]);
    }
    for (int i = sg_local_id + SIMD; i < sub_group_num; i += SIMD) {
      mean = bin_op(mean, static_cast<accscalar_t>(local_mean[i]));
      rstd = bin_op(rstd, static_cast<accscalar_t>(local_rstd[i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      mean = bin_op(mean, static_cast<accscalar_t>(sg.shuffle_down(mean, i)));
      rstd = bin_op(rstd, static_cast<accscalar_t>(sg.shuffle_down(rstd, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final result
    if (sg_local_id == 0) {
      local_mean[0] = mean;
      local_rstd[0] = rstd;
    }
  }
  item_id.barrier(dpcpp_local_fence);

  mean = local_mean[0];
  rstd = local_rstd[0];
}

template <
    int SIMD,
    typename accscalar_t,
    typename IndexType,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared,
    typename local_shared_bool>
static void inline norm_global_reduce(
    nd_item_id item_id,
    int workgroup_num_foreach,
    int local_size,
    int sub_group_num,
    accscalar_t& sum1,
    accscalar_t& sum2,
    accscalar_t* scratchpad_ptr,
    int* semaphores_ptr,
    const local_shared& local_mean,
    const local_shared& local_rstd,
    const local_shared_bool& last_workgroup,
    reduce_op bin_op) {
  IndexType local_id = item_id.get_local_id(2);
  IndexType group_id = item_id.get_group(0);
  IndexType group_id_foreach = item_id.get_group(1);

  if (local_id == 0) {
    // [Batch][2][workgroup_num_foreach]
    auto idx = group_id * workgroup_num_foreach * 2 + group_id_foreach;
    scratchpad_ptr[idx] = sum1;
    scratchpad_ptr[workgroup_num_foreach + idx] = sum2;

    dpcpp_atomic_ref_rlx_dev_global_t<int> count(semaphores_ptr[group_id]);
    int prev_groups_finished = count.fetch_add(1, dpcpp_mem_odr_acq_rel);
    last_workgroup[0] = (prev_groups_finished == workgroup_num_foreach - 1);
  }
  item_id.barrier(dpcpp_local_fence);

  // use the last workgroup for reduction
  if (last_workgroup[0]) {
    sum1 = accscalar_t(0);
    sum2 = accscalar_t(0);
    for (int i = local_id; i < workgroup_num_foreach; i += local_size) {
      auto idx = group_id * workgroup_num_foreach * 2 + i;
      sum1 = bin_op(sum1, scratchpad_ptr[idx]);
      sum2 = bin_op(sum2, scratchpad_ptr[workgroup_num_foreach + idx]);
    }
    norm_group_reduce<SIMD, accscalar_t>(
        item_id, sub_group_num, sum1, sum2, local_mean, local_rstd, bin_op);
  }
}

template <typename scalar_t, typename accscalar_t>
static void inline project_and_store(
    int offset,
    accscalar_t sum1,
    accscalar_t sum2,
    int Plane,
    scalar_t* mean_data,
    scalar_t* rstd_data,
    scalar_t eps) {
  accscalar_t scale = 1 / static_cast<accscalar_t>(Plane);
  sum1 *= scale;
  sum2 = sum2 * scale - sum1 * sum1;
  mean_data[offset] = static_cast<scalar_t>(sum1);
  rstd_data[offset] = static_cast<scalar_t>(Numerics<accscalar_t>::rsqrt(
      sum2 < 0 ? 0 : sum2 + static_cast<accscalar_t>(eps)));
}

template <typename accscalar_t, int SIMD>
static void init_scratchpad(
    const Tensor& X,
    Tensor& semaphores,
    Tensor& scratchpad,
    int& sub_group_num,
    int semaphores_size,
    int scratchpad_size,
    int workgroup_num_foreach) {
  semaphores = at::zeros(semaphores_size, X.options().dtype(kInt));

  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  scratchpad = at::zeros(scratchpad_size, X.options().dtype(kAccType));
  sub_group_num =
      std::min((workgroup_num_foreach + SIMD - 1) / SIMD, sub_group_num);
}
} // namespace normalization

} // namespace AtenIpexTypeXPU
} // namespace at
