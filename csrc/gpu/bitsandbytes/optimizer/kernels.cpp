#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include <cfloat>
#include "ops.h"
#include "optimizer.h"

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

typedef at::Half half;
typedef at::BFloat16 bfloat16;

inline void atomicAdd(float& addr, float value, bool shared) {
  if (shared) {
    sycl::atomic_ref<
        float,
        sycl::ext::oneapi::detail::memory_order::relaxed,
        sycl::ext::oneapi::detail::memory_scope::work_group,
        sycl::access::address_space::local_space>
        atomic_local(addr);
    atomic_local.fetch_add(value);
  } else {
    sycl::atomic_ref<
        float,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atomic_res(addr);
    atomic_res.fetch_add(value);
  }
}

template <typename T>
inline void BlockLoad(
    sycl::nd_item<1> item,
    int items_per_thread,
    int valid_items,
    T default_value,
    T* src,
    T* dst) {
  for (int lt = 0; lt < items_per_thread; lt++) {
    if (item.get_local_id(0) * items_per_thread + lt < valid_items) {
      dst[lt] = src[item.get_local_id(0) * items_per_thread + lt];
    } else {
      dst[lt] = (T)default_value;
    }
  }
}

template <typename T>
inline void BlockStore(
    sycl::nd_item<1> item,
    int items_per_thread,
    int valid_items,
    T* src,
    T* dst) {
  for (int lt = 0; lt < items_per_thread; lt++) {
    if (item.get_local_id(0) * items_per_thread + lt < valid_items) {
      dst[item.get_local_id(0) * items_per_thread + lt] = src[lt];
    }
  }
}

inline float BlockReduce(
    sycl::nd_item<1> item,
    float input,
    int op = 1,
    int valid_items = 1024) {
  if (item.get_local_id(0) >= valid_items)
    input = 0.0f;
  if (op == 1) {
    return sycl::reduce_over_group(item.get_group(), input, sycl::plus<>());
  } else if (op == 2) {
    return sycl::reduce_over_group(
        item.get_group(), input, sycl::maximum<float>());
  }
}

inline bool sycl_signbit(float x) {
  return x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f);
}

template <int SIGNED>
inline unsigned char quantize_2D(
    float* quadrants,
    float* const smem_code,
    float x) {
  int pivot = 127;
  int upper_pivot = 255;
  int lower_pivot = 0;

  float lower = SIGNED ? -1.0f : 0.0f;
  float upper = 1.0f;
  float midpoint;
  float val = quadrants[1];
  int local_pivot = 1;
  int offset = 1;

  // i>>=1 = {32, 16, 8, 4, 2, 1}
  for (int i = 64; i > 0; i >>= 1) {
    if (x > val) {
      lower_pivot = pivot;
      lower = val;
      pivot += i;
      // val = i == 64 ? quadrants[2] : smem_code[pivot];
      local_pivot += offset;
    } else {
      upper_pivot = pivot;
      upper = val;
      pivot -= i;
      // val = i == 64 ? quadrants[0] : smem_code[pivot];
      local_pivot -= offset;
    }
    val = i >= 64 ? quadrants[local_pivot] : smem_code[pivot];
    offset -= 1;
  }

  if (x > val) {
    midpoint = (upper + val) * 0.5f;
    if (x > midpoint)
      return upper_pivot;
    else
      return pivot;
  } else {
    midpoint = (lower + val) * 0.5f;
    if (x < midpoint)
      return lower_pivot;
    else
      return pivot;
  }
}

template <typename T, int BLOCK_SIZE, int NUM_VALS>
SYCL_EXTERNAL void kPercentileClipping<T, BLOCK_SIZE, NUM_VALS>::operator()(
    sycl::nd_item<1> item) const {
  sycl::group<1> grp = item.get_group();

  const int n_full =
      (BLOCK_SIZE * (n / BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  int valid_items = 0;
  T vals[NUM_VALS];
  float local_sum = 0.0f;
  auto& shared_data =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
          sycl::ext::oneapi::experimental::this_group<1>());
  shared_data[0] = 0.0f;

  for (unsigned int i = (item.get_group(0) * BLOCK_SIZE); i < n_full;
       i += item.get_group_range(0) * BLOCK_SIZE) {
    valid_items = (n - i) > BLOCK_SIZE ? BLOCK_SIZE : (n - i);
    local_sum = 0.0f;

    item.barrier();
    BlockLoad<T>(item, NUM_VALS, valid_items, (T)0.0f, &g[i], vals);

#pragma unroll NUM_VALS
    for (int j = 0; j < NUM_VALS; j++)
      local_sum += ((float)vals[j]) * ((float)vals[j]);
    shared_data[0] = BlockReduce(item, local_sum, 1, valid_items);

    if (item.get_local_id(0) == 0) {
      if (step == 1) {
        // initialize with the same norm for all positions
        //#pragma unroll 10
        for (int j = 0; j < 100; j++) {
          atomicAdd(gnorm_vec[j], shared_data[0], false);
        }

      } else
        atomicAdd(gnorm_vec[step % 100], shared_data[0], false);
    }
  }
}
template class kPercentileClipping<float, 2048, 4>;
template class kPercentileClipping<half, 2048, 4>;

#define LANES 2
#define QUAD 3
template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
SYCL_EXTERNAL void kOptimizerStatic8bit2StateBlockwise<
    T,
    OPTIMIZER,
    BLOCK_SIZE,
    N_PER_TH>::operator()(sycl::nd_item<1> item) const {
  sycl::group<1> grp = item.get_group();

  const int n_full = item.get_group_range(0) * BLOCK_SIZE;
  const int base_idx = (item.get_group(0) * BLOCK_SIZE);
  int valid_items = 0;
  float g_val = 0.0f;
  float s1_vals[N_PER_TH];
  float s2_vals[N_PER_TH];
  float s3_vals[N_PER_TH];

  // 2-5%
  const float correction1 = 1.0f - sycl::pow(beta1, step);
  const float correction2 = sycl::sqrt(1.0f - sycl::pow(beta2, step));
  const float step_size = (-lr * correction2 / correction1);
  const int lane_id = item.get_local_id(0) % LANES;
  float new_local_abs_max1 = -FLT_MAX;
  float new_local_abs_max2 = -FLT_MAX;
  float new_local_abs_max3 = -FLT_MAX;
  float quadrants1[QUAD];
  float quadrants2[QUAD];

  unsigned char c1s[N_PER_TH];
  unsigned char c2s[N_PER_TH];
  unsigned char c3s[N_PER_TH];

  T g_vals[N_PER_TH];
  T p_vals[N_PER_TH];

  auto& smem_quantiles1 =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[LANES][257]>(
          sycl::ext::oneapi::experimental::this_group<1>());
  auto& smem_quantiles2 =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[LANES][257]>(
          sycl::ext::oneapi::experimental::this_group<1>());

  auto& smem_exchange1 =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
          sycl::ext::oneapi::experimental::this_group<1>());
  auto& smem_exchange2 =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
          sycl::ext::oneapi::experimental::this_group<1>());
  auto& smem_exchange3 =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
          sycl::ext::oneapi::experimental::this_group<1>());

  // 0.23 -> 0.23
  smem_quantiles1[0][item.get_local_id(0)] = quantiles1[item.get_local_id(0)];
  smem_quantiles2[0][item.get_local_id(0)] = quantiles2[item.get_local_id(0)];
#pragma unroll
  for (unsigned int j = 1; j < LANES; j++) {
    smem_quantiles1[j][item.get_local_id(0)] =
        smem_quantiles1[0][item.get_local_id(0)];
    smem_quantiles2[j][item.get_local_id(0)] =
        smem_quantiles2[0][item.get_local_id(0)];
  }
  item.barrier();

#pragma unroll
  for (int k = 0; k < QUAD; k++) {
    quadrants1[k] =
        smem_quantiles1[lane_id]
                       [(k * 256 / (QUAD + 1)) + (256 / (QUAD + 1) - 1)];
    quadrants2[k] =
        smem_quantiles2[lane_id]
                       [(k * 256 / (QUAD + 1)) + (256 / (QUAD + 1) - 1)];
  }

  for (unsigned int i = base_idx; i < n_full;
       i += item.get_group_range(0) * BLOCK_SIZE) {
    // loads: 0.23 -> 0.85/1.44
    valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
    item.barrier();
    BlockLoad<T>(item, N_PER_TH, valid_items, (T)0.0f, &(g[i]), g_vals);
    item.barrier();
    BlockLoad<unsigned char>(
        item, N_PER_TH, valid_items, 128, &(state1[i]), c1s);
    item.barrier();
    BlockLoad<unsigned char>(
        item, N_PER_TH, valid_items, 128, &(state2[i]), c2s);

    new_local_abs_max1 = -FLT_MAX;
    new_local_abs_max2 = -FLT_MAX;
    new_local_abs_max3 = -FLT_MAX;

    //  update: 2.48/1.57 -> 2.51/1.60
    for (unsigned int j = 0; j < N_PER_TH; j++) {
      if (!sycl::isnan((float)g_vals[j]) && !sycl::isinf((float)g_vals[j])) {
        s2_vals[j] = smem_quantiles2[lane_id][c2s[j]] * absmax2[i / BLOCK_SIZE];
        g_val = g_vals[j];
        g_val *= gnorm_scale;

        s2_vals[j] = (s2_vals[j] * beta2) + (((1.0f - beta2) * g_val * g_val));

        s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * absmax1[i / BLOCK_SIZE];
        s1_vals[j] = (s1_vals[j] * beta1) + (((1.0f - beta1) * g_val));

      } else {
        s1_vals[j] = 0.0f;
        s2_vals[j] = 0.0f;
      }

      new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
      new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));
    }

    //  reduce: 2.51/1.60 -> 2.67/1.69
    new_local_abs_max1 = BlockReduce(item, new_local_abs_max1, 2);
    new_local_abs_max2 = BlockReduce(item, new_local_abs_max2, 2);

    if (item.get_local_id(0) == 0) {
      smem_exchange1[0] = new_local_abs_max1;
      smem_exchange2[0] = new_local_abs_max2;
    }

    item.barrier();

    if (item.get_local_id(0) == 0) {
      absmax1[i / BLOCK_SIZE] = new_local_abs_max1;
      absmax2[i / BLOCK_SIZE] = new_local_abs_max2;
    } else {
      new_local_abs_max1 = smem_exchange1[0];
      new_local_abs_max2 = smem_exchange2[0];
    }
    item.barrier();

    BlockLoad<T>(item, N_PER_TH, valid_items, (T)0.0f, &(p[i]), p_vals);

    //  reduce: 2.67/1.69 -> 2.67/1.70
    for (unsigned int j = 0; j < N_PER_TH; j++) {
      if (!sycl::isnan((float)g_vals[j]) && !sycl::isinf((float)g_vals[j])) {
        p_vals[j] =
            (T)(((float)p_vals[j]) +
                ((step_size *
                  (s1_vals[j] / (sqrtf(s2_vals[j]) + (correction2 * eps))))));

        if (weight_decay > 0.0f)
          p_vals[j] = ((float)p_vals[j]) * (1.0f - (lr * weight_decay));
      }
    }

    //  store: 0.85/1.44 -> 2.48/1.57
    item.barrier();
    BlockStore<T>(item, N_PER_TH, valid_items, p_vals, &(p[i]));

    //  quantizaztion: 2.67/1.70  -> 3.4/3.3
    for (unsigned int j = 0; j < N_PER_TH; j++) {
      c1s[j] = quantize_2D<1>(
          quadrants1,
          smem_quantiles1[lane_id],
          (s1_vals[j] / new_local_abs_max1));
      c2s[j] = quantize_2D<0>(
          quadrants2,
          smem_quantiles2[lane_id],
          (s2_vals[j] / new_local_abs_max2));

      // make sure state1 term has still the same sign after quantization
      // (not needed for state2 term which has only positive values)
      if (sycl_signbit(smem_quantiles1[lane_id][c1s[j]]) !=
          sycl_signbit(s1_vals[j])) {
        if (s1_vals[j] > 0.0f)
          c1s[j] += 1;
        else
          c1s[j] -= 1;
      }
    }

    item.barrier();
    BlockStore<unsigned char>(item, N_PER_TH, valid_items, c1s, &(state1[i]));
    item.barrier();
    BlockStore<unsigned char>(item, N_PER_TH, valid_items, c2s, &(state2[i]));
  }
}
template class kOptimizerStatic8bit2StateBlockwise<float, ADAM, 256, 1>;
template class kOptimizerStatic8bit2StateBlockwise<half, ADAM, 256, 1>;
template class kOptimizerStatic8bit2StateBlockwise<bfloat16, ADAM, 256, 1>;

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
SYCL_EXTERNAL void kDequantizeBlockwise<
    T,
    TILE_SIZE,
    THREADS,
    NUM_PER_TH,
    DATA_TYPE>::operator()(sycl::nd_item<1> item) const {
  const int n_load = (item.get_group_range(0) * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (item.get_group(0) * TILE_SIZE);

  T vals[NUM_PER_TH * ((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;

  for (int i = base_idx; i < n_load; i += item.get_group_range(0) * TILE_SIZE) {
    if (DATA_TYPE > 0) {
      valid_items_load = sycl::min(TILE_SIZE, (n + 1) / 2 - i);
      valid_items_store = sycl::min(TILE_SIZE * 2, n - i * 2);
    } else {
      valid_items_load = sycl::min(TILE_SIZE, n - i);
      valid_items_store = valid_items_load;
    }

    local_abs_max = absmax[(i + item.get_local_id(0) * NUM_PER_TH) / blocksize];

    item.barrier();
    BlockLoad<unsigned char>(
        item, NUM_PER_TH, valid_items_load, 128, &(A[i]), qvals);

    switch (DATA_TYPE) {
      case General8bit:
// load code through read-only cache via __ldg
#pragma unroll NUM_PER_TH
        for (int j = 0; j < NUM_PER_TH; j++)
          vals[j] = code[qvals[j]] * local_abs_max;
        break;
    }

    item.barrier();
    BlockStore<T>(
        item,
        NUM_PER_TH,
        valid_items_store,
        vals,
        &(out[(DATA_TYPE > 0) ? i * 2 : i]));
  }
}
template class kDequantizeBlockwise<float, 512, 64, 8, General8bit>;
template class kDequantizeBlockwise<half, 512, 64, 8, General8bit>;
template class kDequantizeBlockwise<bfloat16, 512, 64, 8, General8bit>;