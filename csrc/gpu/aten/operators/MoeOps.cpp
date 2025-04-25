#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <CL/sycl.hpp>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "utils/ComputeEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

#define MaxTopK 2
#define MaxNumExperts 32

namespace TopKSoftmaxImpl {
// Each WI compute  one token
template <int TopK, typename T>
struct FusedTopkSoftmax {
  static constexpr int SgSize = 16;
  static constexpr int MaxSg = 64;
  static constexpr float kNegInfinity = INFINITY * -1;
  static constexpr int ElemsPerItem = sizeof(T) > sizeof(float)
      ? 1
      : sizeof(float) / sizeof(T);

  FusedTopkSoftmax(
      sycl::local_accessor<T, 1> slm,
      const T* gating_output,
      float* topk_weights,
      int* topk_indices,
      int* rows_for_experts,
      int* offsets,
      const int num_tokens,
      const int num_experts)
      : slm(slm),
        gating_output(gating_output),
        topk_weights(topk_weights),
        topk_indices(topk_indices),
        rows_for_experts(rows_for_experts),
        offsets(offsets),
        num_tokens(num_tokens),
        num_experts(num_experts) {}

  static inline int get_slm_size(const int num_tokens, const int num_experts) {
    int total_sg = (num_tokens + SgSize - 1) / SgSize;
    int local_sg = total_sg < MaxSg ? total_sg : MaxSg;
    return local_sg * SgSize * num_experts * sizeof(T);
  }

  static inline sycl::nd_range<3> get_nd_range(const int num_tokens) {
    int total_sg = (num_tokens + SgSize - 1) / SgSize;
    int local_sg = total_sg < MaxSg ? total_sg : MaxSg;
    int num_wg = (total_sg + local_sg - 1) / local_sg;
    sycl::range<3> local(1, 1, local_sg * SgSize);
    sycl::range<3> global(1, 1, num_wg);
    return sycl::nd_range<3>(global * local, local);
  }

  void operator()(sycl::nd_item<3> item) const {
    T* slm_ptr =
        slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    // load data from global memory to shared memory
    int local_range = item.get_local_range(2);
    int gid = item.get_group(2);
    int group_offset = gid * local_range;

    if (group_offset >= num_tokens)
      return;

    int local_id = item.get_local_linear_id();

    int start = group_offset * num_experts;
    int end = (group_offset + local_range) < num_tokens
        ? (group_offset + local_range) * num_experts
        : num_tokens * num_experts;
    // TODO(Performance): fix instruction dependency
    for (int i = local_id * ElemsPerItem; (start + i) < end;
         i += local_range * ElemsPerItem) {
      T logits[ElemsPerItem];
#pragma unroll
      for (int j = 0; j < ElemsPerItem; j++) {
        logits[j] = *(gating_output + start + i + j);
      }

      for (int j = 0; j < ElemsPerItem; j++) {
        slm_ptr[i + j] = logits[j];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (group_offset + local_id >= num_tokens)
      return;

    // calculate topk
    sycl::vec<float, TopK> topk_data(kNegInfinity);
    sycl::vec<int, TopK> topk_idx(-1);
    int local_offset = local_id * num_experts;
    for (int k = 0; k < TopK; ++k) {
      for (int i = 0; i < num_experts / ElemsPerItem; ++i) {
        T data[ElemsPerItem];
        for (int j = 0; j < ElemsPerItem; ++j) {
          data[j] = slm_ptr[local_offset + i * ElemsPerItem + j];
        }

        for (int j = 0; j < ElemsPerItem; ++j) {
          if (data[j] > topk_data[k]) {
            topk_data[k] = data[j];
            topk_idx[k] = i * ElemsPerItem + j;
          }
        }
      }
      slm_ptr[local_offset + topk_idx[k]] = kNegInfinity;
    }

    // perform softmax
    const float softmax_max = topk_data[0];
    float softmax_sum(0);
    for (int k = 0; k < TopK; ++k) {
      topk_data[k] = sycl::exp(topk_data[k] - softmax_max);
      softmax_sum += topk_data[k];
    }

    for (int i = 0; i < num_experts / ElemsPerItem; ++i) {
      T data[ElemsPerItem];
      for (int j = 0; j < ElemsPerItem; ++j) {
        data[j] = slm_ptr[local_offset + i * ElemsPerItem + j];
      }
      for (int j = 0; j < ElemsPerItem; ++j)
        softmax_sum += sycl::exp(float(data[j]) - softmax_max);
    }

    for (int k = 0; k < TopK; ++k) {
      topk_data[k] /= softmax_sum;
    }

    // store data to global memory and  atomic add to rows_for_experts
    // TODO: add hierarchy atomic add for large token number senario
    int offset = group_offset * TopK + local_id * TopK;
    for (int k = 0; k < TopK; ++k) {
      topk_weights[offset + k] = topk_data[k];
      topk_indices[offset + k] = topk_idx[k];
      auto ref_num_tokens = sycl::atomic_ref<
          int,
          sycl::memory_order_relaxed,
          sycl::memory_scope_device,
          sycl::access::address_space::global_space>(
          *(rows_for_experts + topk_idx[k]));
      int old = ref_num_tokens.fetch_add(1);
      offsets[offset + k] = old;
    }
  }
  sycl::local_accessor<T, 1> slm;
  const T* gating_output;
  float* topk_weights;
  int* topk_indices;
  int* rows_for_experts;
  int* offsets;
  const int num_tokens;
  const int num_experts;
};
template <int TopK, typename T>
void launch_fused_topk_softmax(
    sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    int* rows_for_experts,
    int* offsets,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSoftmax<TopK, T>;
  int slm_size = Kernel::get_slm_size(num_tokens, num_experts);
  auto range = Kernel::get_nd_range(num_tokens);
  auto cgf = DPCPP_Q_CGF(cgh) {
    sycl::local_accessor<T, 1> accessor(slm_size / sizeof(T), cgh);
    Kernel task(
        accessor,
        gating_output,
        topk_weights,
        topk_indices,
        rows_for_experts,
        offsets,
        num_tokens,
        num_experts);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T>
void fused_topk_softmax(
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    int* rows_for_experts,
    int* offsets,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  auto& queue = dpcppGetCurrentQueue();

  TORCH_CHECK(
      num_experts <= MaxNumExperts && MaxNumExperts % num_experts == 0,
      "num_experts must be less than or equal to 32 and 32 must be divisible by num_experts");

  switch (topk) {
    case 1:
      launch_fused_topk_softmax<1, T>(
          queue,
          gating_output,
          topk_weights,
          topk_indices,
          rows_for_experts,
          offsets,
          num_tokens,
          num_experts);
      break;
    case 2:
      launch_fused_topk_softmax<2, T>(
          queue,
          gating_output,
          topk_weights,
          topk_indices,
          rows_for_experts,
          offsets,
          num_tokens,
          num_experts);
      break;
    default:
      TORCH_CHECK(
          false, "error: not support topk=%d, up to topk=%d\n", topk, MaxTopK);
  };
};
}; // namespace TopKSoftmaxImpl

namespace GroupedTopKImpl {
template <typename T, bool addbias, bool renorm, int topk_group, int topk>
struct FusedGroupedTopK {
  static constexpr float kPosInfinity = INFINITY;
  static constexpr float kNegInfinity = (-1) * kPosInfinity;
  static constexpr int SgSize = 16;
  using VECBIT = sycl::vec<uint8_t, MaxNumExperts / 8>;
  FusedGroupedTopK(
      const T* gating_output,
      const T* bias,
      float* topk_weights,
      int* topk_indices,
      float* score_buf,
      float* max_buf,
      const int num_tokens,
      const int num_experts,
      const int num_expert_group)
      : gating_output(gating_output),
        bias(bias),
        topk_weights(topk_weights),
        topk_indices(topk_indices),
        score_buf(score_buf),
        max_buf(max_buf),
        num_tokens(num_tokens),
        num_experts(num_experts),
        num_expert_group(num_expert_group),
        chk_per_group(num_experts / num_expert_group) {}

  static inline sycl::nd_range<1> get_nd_range(const int num_tokens) {
    int total_sg = divup(num_tokens, SgSize) * SgSize;
    int local_sg = SgSize;
    sycl::range<1> local(local_sg);
    sycl::range<1> global(total_sg);
    return sycl::nd_range<1>(global, local);
  }

  inline static void bit_set(uint8_t& c, int b) {
    c |= (1 << b);
  }

  inline static void bit_clear(uint8_t& c, int b) {
    c &= (~(1 << b));
  }

  inline static int bit_test(uint8_t& c, int b) {
    return (c & (1 << b)) ? 1 : 0;
  }

  inline static int bit_wordaddr(int b) {
    return b >> 3;
  }

  inline static int bit_bitaddr(int b) {
    return b & 7;
  }

  inline static void vec_set(VECBIT& mask, int i) {
    bit_set(mask[bit_wordaddr(i)], bit_bitaddr(i));
  }
  inline static void vec_clear(VECBIT& mask, int i) {
    bit_clear(mask[bit_wordaddr(i)], bit_bitaddr(i));
  }

  inline static int vec_test(VECBIT& mask, int i) {
    return bit_test(mask[bit_wordaddr(i)], bit_bitaddr(i));
  }

  template <int n>
  inline void get_topk_sort(
      sycl::vec<float, n>& val,
      sycl::vec<int, n>& idx,
      float* src,
      int dim) const {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < dim; ++j) {
        if (val[i] < src[j]) {
          idx[i] = j;
          val[i] = src[j];
        }
      }
      src[idx[i]] = kNegInfinity;
    }
  }

  template <int n>
  inline int argmin(sycl::vec<float, n>& val) const {
    float min_val = kPosInfinity;
    int idx = -1;
    for (int i = 0; i < n; ++i) {
      if (min_val > val[i]) {
        min_val = val[i];
        idx = i;
      }
    }
    return idx;
  }

  template <int n>
  inline void get_topk_nosort(
      sycl::vec<float, n>& val,
      sycl::vec<int, n>& idx,
      float* src,
      int dim) const {
    float min_val = kPosInfinity;
    int min_idx = -1;
    for (int i = 0; i < n; ++i) {
      val[i] = src[i];
      idx[i] = i;
      if (src[i] < min_val) {
        min_val = src[i];
        min_idx = i;
      }
    }
    for (int i = n; i < dim; ++i) {
      if (src[i] > min_val) {
        val[min_idx] = src[i];
        idx[min_idx] = i;
        int j = argmin(val);
        min_idx = j;
        min_val = val[j];
      }
    }
  }

  void grouped_topk_kern(
      const T* src,
      float* topk_wgts,
      int* topk_ids,
      const T* bias,
      float* max_buf,
      float* score_buf,
      const int64_t num_token,
      const int64_t inner_dim,
      const int64_t chk_per_group,
      const int64_t num_expert_group) const {
    // src num_token, inner_dim
    // dst num_token, num_expert_group
    // no support softmax
    float one = 1.0f;
    float max_val;
    // sigmoid, addbias and max
    for (int64_t i = 0; i < num_expert_group; ++i) {
      max_val = kNegInfinity;
      for (int64_t j = 0; j < chk_per_group; ++j) {
        int64_t k = i * chk_per_group + j;
        float t;
        if constexpr (addbias) {
          t = one / (one + sycl::native::exp(-src[k])) + bias[k];
        } else {
          t = one / (one + sycl::native::exp(-src[k]));
        }
        score_buf[k] = t;
        max_val = std::max(t, max_val);
      }
      max_buf[i] = max_val; // num_token, num_expert_group
    }
    // topk index only
    sycl::vec<float, topk_group> topk_group_data(kNegInfinity);
    sycl::vec<int, topk_group> topk_group_idx(-1);
    get_topk_nosort(topk_group_data, topk_group_idx, max_buf, num_expert_group);
    // reverse mask
    VECBIT mask(0xff); // 32 max_expert_group, 8 for each uchar
    for (int i = 0; i < topk_group; ++i) {
      vec_clear(mask, topk_group_idx[i]);
    }
    // fill with 0
    for (int i = 0; i < num_expert_group; ++i) {
      if (vec_test(mask, i)) {
        int s = i * chk_per_group;
        for (int j = s; j < s + chk_per_group; ++j) {
          score_buf[j] = 0.0f;
        }
      }
    }
    // final topk
    sycl::vec<float, topk> topk_data(kNegInfinity);
    sycl::vec<int, topk> topk_idx(-1);
    get_topk_nosort(topk_data, topk_idx, score_buf, inner_dim);

    float row_sum = 0.0f;
    for (int i = 0; i < topk; ++i) {
      row_sum += topk_data[i];
    }
    for (int i = 0; i < topk; ++i) {
      if constexpr (renorm) {
        topk_wgts[i] = topk_data[i] / row_sum;
      } else {
        topk_wgts[i] = topk_data[i];
      }
      topk_ids[i] = topk_idx[i];
    }
  }

  void operator()(sycl::nd_item<1> it) const {
    int64_t token_id = it.get_global_id()[0];
    int64_t i = token_id * num_experts;
    int64_t j = token_id * num_expert_group;
    int64_t k = token_id * topk;
    if (token_id < num_tokens) {
      grouped_topk_kern(
          &gating_output[i],
          &topk_weights[k],
          &topk_indices[k],
          bias,
          &max_buf[j],
          &score_buf[i],
          num_tokens,
          num_experts,
          chk_per_group,
          num_expert_group);
    }
  }

  const T* gating_output;
  const T* bias;
  float* topk_weights;
  int* topk_indices;
  float* score_buf;
  float* max_buf;
  const int num_tokens;
  const int num_experts;
  const int num_expert_group;
  const int64_t chk_per_group;
};

template <typename T, int topk, int topk_group>
void launch_fused_grouped_topk_sigmoid(
    sycl::queue& queue,
    const T* gating_output,
    const T* bias,
    float* topk_weights,
    int* topk_indices,
    float* score_buf,
    float* max_buf,
    const int num_tokens,
    const int num_experts,
    const int num_expert_group) {
  // addbias and renorm are set to true by default
  using Kernel = FusedGroupedTopK<T, true, true, topk_group, topk>;
  // int slm_size = Kernel::get_slm_size(num_tokens, num_experts);
  auto range = Kernel::get_nd_range(num_tokens);
  auto cgf = DPCPP_Q_CGF(cgh) {
    // sycl::local_accessor<T, 1> accessor(slm_size / sizeof(T), cgh);
    Kernel task(
        // accessor,
        gating_output,
        bias,
        topk_weights,
        topk_indices,
        score_buf,
        max_buf,
        num_tokens,
        num_experts,
        num_expert_group);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T>
using LAUNCH_FUNC = void (*)(
    sycl::queue&,
    const T*,
    const T*,
    float*,
    int*,
    float*,
    float*,
    const int,
    const int,
    const int);

#define DEFINE_GROUPED_TOPK_FUNC(T, a, b) \
  &launch_fused_grouped_topk_sigmoid<T, a, b>

template <typename T>
void fused_grouped_topk_sigmoid(
    const T* gating_output,
    const T* bias,
    float* topk_weights,
    int* topk_indices,
    float* score_buf,
    float* max_buf,
    const int num_tokens,
    const int num_experts,
    const int num_expert_group,
    const int topk,
    const int topk_group) {
  auto& queue = dpcppGetCurrentQueue();

  TORCH_CHECK(
      num_expert_group <= MaxNumExperts,
      "num_experts must be less than or equal to 32");
  TORCH_CHECK(
      topk_group <= num_expert_group,
      "topk_group must be less than or equal to num_expert_group");
  constexpr std::array<int, 5> allowed_k = {1, 2, 4, 8, 16};
  int topk_idx = -1;
  int topk_g_idx = -1;
  for (int i = 0; i < allowed_k.size(); ++i) {
    if (allowed_k[i] == topk) {
      topk_idx = i;
    }
    if (allowed_k[i] == topk_group) {
      topk_g_idx = i;
    }
  }
  TORCH_CHECK(
      topk_idx >= 0 && topk_g_idx >= 0,
      "wrong values for topk (%d) and topk_group (%d) up to %d\n",
      topk,
      topk_group,
      16);
  int funcIndex = topk_idx * allowed_k.size() + topk_g_idx;
  static constexpr std::array<LAUNCH_FUNC<T>, 25> launch_funcs = {
      DEFINE_GROUPED_TOPK_FUNC(T, 1, 1),   DEFINE_GROUPED_TOPK_FUNC(T, 1, 2),
      DEFINE_GROUPED_TOPK_FUNC(T, 1, 4),   DEFINE_GROUPED_TOPK_FUNC(T, 1, 8),
      DEFINE_GROUPED_TOPK_FUNC(T, 1, 16),  DEFINE_GROUPED_TOPK_FUNC(T, 2, 1),
      DEFINE_GROUPED_TOPK_FUNC(T, 2, 2),   DEFINE_GROUPED_TOPK_FUNC(T, 2, 4),
      DEFINE_GROUPED_TOPK_FUNC(T, 2, 8),   DEFINE_GROUPED_TOPK_FUNC(T, 2, 16),
      DEFINE_GROUPED_TOPK_FUNC(T, 4, 1),   DEFINE_GROUPED_TOPK_FUNC(T, 4, 2),
      DEFINE_GROUPED_TOPK_FUNC(T, 4, 4),   DEFINE_GROUPED_TOPK_FUNC(T, 4, 8),
      DEFINE_GROUPED_TOPK_FUNC(T, 4, 16),  DEFINE_GROUPED_TOPK_FUNC(T, 8, 1),
      DEFINE_GROUPED_TOPK_FUNC(T, 8, 2),   DEFINE_GROUPED_TOPK_FUNC(T, 8, 4),
      DEFINE_GROUPED_TOPK_FUNC(T, 8, 8),   DEFINE_GROUPED_TOPK_FUNC(T, 8, 16),
      DEFINE_GROUPED_TOPK_FUNC(T, 16, 1),  DEFINE_GROUPED_TOPK_FUNC(T, 16, 2),
      DEFINE_GROUPED_TOPK_FUNC(T, 16, 4),  DEFINE_GROUPED_TOPK_FUNC(T, 16, 8),
      DEFINE_GROUPED_TOPK_FUNC(T, 16, 16),
  };

  launch_funcs[funcIndex](
      queue,
      gating_output,
      bias,
      topk_weights,
      topk_indices,
      score_buf,
      max_buf,
      num_tokens,
      num_experts,
      num_expert_group);
}

}; // namespace GroupedTopKImpl

namespace MoEScatterImpl {

/**
 * @brief Performs sub-group inclusive scan operation.
 * @tparam ScanT The data type of the value to scan.
 * @param item The SYCL nd_item object.
 * @param value The value to scan.
 * @return The scanned value.
 */
template <typename ScanT, int SGSize>
inline ScanT sub_group_inclusive_scan(sycl::nd_item<1> item, ScanT value) {
  auto sg = item.get_sub_group();
  auto sg_id = sg.get_group_linear_id();
  auto lane_id = sg.get_local_linear_id();
  for (int i = 1; i < SGSize; i *= 2) {
    auto tmp = sycl::shift_group_right(sg, value, i);
    if (lane_id >= i)
      value += tmp;
  }
  return value;
}

/**
 * @brief Performs group inclusive scan operation.
 * @tparam ScanT The data type of the value to scan.
 * @param item The SYCL nd_item object.
 * @param value The value to scan.
 * @return The scanned value.
 */
template <typename ScanT, int NumSg, int SgSize>
inline ScanT group_inclusive_scan(sycl::nd_item<1> item, ScanT value) {
  auto sg = item.get_sub_group();
  auto sg_id = sg.get_group_linear_id();
  auto lane_id = sg.get_local_linear_id();
  sycl::multi_ptr<int[NumSg], sycl::access::address_space::local_space>
      across_warp_cumsum = sycl::ext::oneapi::
          group_local_memory_for_overwrite<int[NumSg], sycl::group<1>>(
              item.get_group());
  auto& across_warp_cumsum_ptr = *across_warp_cumsum;

  // subgroup wide prefix sum
  for (int i = 1; i < SgSize; i *= 2) {
    auto tmp = sycl::shift_group_right(sg, value, i);
    if (lane_id >= i)
      value += tmp;
  }
  if (lane_id == SgSize - 1)
    across_warp_cumsum_ptr[sg_id] = value;
  item.barrier(sycl::access::fence_space::local_space);

  int sg_prefix;
  int prefix = 0;
  for (int i = 0; i < NumSg; ++i) {
    if (sg_id == i)
      sg_prefix = prefix;
    prefix += across_warp_cumsum_ptr[i];
  }
  return value + sg_prefix;
}

template <int TopK, typename T>
struct MoEScatter {
  static constexpr int GroupWorkItem = 256;
  static constexpr int SgSize = 16;
  static constexpr int NumSg = GroupWorkItem / SgSize;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);

  MoEScatter(
      const T* activations,
      const int* rows_for_experts,
      const int* topk_indices,
      const int* offsets,
      T* reordered_activation,
      int* mapped_slot,
      const int n_tokens,
      const int n_experts,
      const int n_channels)
      : activations(activations),
        rows_for_experts(rows_for_experts),
        topk_indices(topk_indices),
        offsets(offsets),
        reordered_activation(reordered_activation),
        mapped_slot(mapped_slot),
        n_tokens(n_tokens),
        n_experts(n_experts),
        n_channels(n_channels) {}

  /**
   * @brief Returns the 1D range for the kernel execution.
   * @param n_tokens The number of tokens.
   * @return The 1D range for the kernel execution.
   */
  static inline sycl::nd_range<1> get_nd_range(const int n_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(n_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  /**
   * @brief The kernel operator.
   * @param item The SYCL nd_item object.
   */
  [[sycl::reqd_sub_group_size(SgSize)]] void operator()(
      sycl::nd_item<1> item) const {
    // perform prefix sum on rows_for_experts to get expert offset
    auto token_id = item.get_group(0);
    auto local_id = item.get_local_linear_id();

    // allocate shared memory for expert index of current token
    sycl::
        multi_ptr<int[GroupWorkItem], sycl::access::address_space::local_space>
            expert_cumsum = sycl::ext::oneapi::group_local_memory_for_overwrite<
                int[GroupWorkItem],
                sycl::group<1>>(item.get_group());
    auto& expert_cumsum_ptr = *expert_cumsum;

    int expert_val = 0;
    if (local_id < n_experts)
      expert_val = rows_for_experts[local_id];

    if (n_experts <= SgSize)
      expert_val = sub_group_inclusive_scan<int, SgSize>(item, expert_val);
    else
      expert_val = group_inclusive_scan<int, NumSg, SgSize>(item, expert_val);

    if (local_id < n_experts)
      expert_cumsum_ptr[local_id] = expert_val;
    item.barrier(sycl::access::fence_space::local_space);

    int indices[TopK], expert_local_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      indices[i] = topk_indices[token_id * TopK + i];
      expert_local_offset[i] = offsets[token_id * TopK + i];
    }

    int expert_row_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      int expert_id = indices[i];
      int start_offset = expert_id == 0 ? 0 : expert_cumsum_ptr[expert_id - 1];
      expert_row_offset[i] = start_offset + expert_local_offset[i];
    }

    const T* activation_base =
        activations + token_id * n_channels + local_id * ElemsPerItem;
    T* reordered_activation_bases[TopK];
    for (int i = 0; i < TopK; ++i) {
      reordered_activation_bases[i] = reordered_activation +
          expert_row_offset[i] * n_channels + local_id * ElemsPerItem;
    }

    constexpr int stride = GroupWorkItem * ElemsPerItem;
    const int loop_count = (n_channels + stride - 1) / stride;

    for (int loop = 0; loop < loop_count; ++loop) {
      using load_type = sycl::vec<T, ElemsPerItem>;
      load_type data;
      if (loop * stride + local_id * ElemsPerItem < n_channels) {
        data = *(reinterpret_cast<const load_type*>(
            activation_base + loop * stride));
        for (int i = 0; i < TopK; ++i) {
          *(reinterpret_cast<load_type*>(
              reordered_activation_bases[i] + loop * stride)) = data;
        }
      }
    }
    if (local_id == 0) {
      for (int i = 0; i < TopK; ++i) {
        mapped_slot[token_id * TopK + i] = expert_row_offset[i];
      }
    }
  }
  const T* activations; // [n_tokens, n_channels]
  const int* rows_for_experts; // [n_experts]
  const int* topk_indices; // [n_tokens, num_top_k]
  const int* offsets; // [n_tokens, num_top_k]
  T* reordered_activation;
  int* mapped_slot; // [n_tokens, num_top_k]
  const int n_tokens;
  const int n_experts;
  const int n_channels;
};

template <int TopK>
struct MoEScatter<TopK, sycl::ext::oneapi::bfloat16> {
  using T = sycl::ext::oneapi::bfloat16;
  static constexpr int GroupWorkItem = 256;
  static constexpr int SgSize = 16;
  static constexpr int NumSg = GroupWorkItem / SgSize;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  MoEScatter(
      const T* activations,
      const int* rows_for_experts,
      const int* topk_indices,
      const int* offsets,
      T* reordered_activation,
      int* mapped_slot,
      const int n_tokens,
      const int n_experts,
      const int n_channels)
      : activations(activations),
        rows_for_experts(rows_for_experts),
        topk_indices(topk_indices),
        offsets(offsets),
        reordered_activation(reordered_activation),
        mapped_slot(mapped_slot),
        n_tokens(n_tokens),
        n_experts(n_experts),
        n_channels(n_channels) {}

  /**
   * @brief Returns the 1D range for the kernel execution.
   * @param n_tokens The number of tokens.
   * @return The 1D range for the kernel execution.
   */
  static inline sycl::nd_range<1> get_nd_range(const int n_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(n_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  /**
   * @brief The kernel operator.
   * @param item The SYCL nd_item object.
   */
  [[sycl::reqd_sub_group_size(SgSize)]] void operator()(
      sycl::nd_item<1> item) const {
    // perform prefix sum on rows_for_experts to get expert offset
    auto token_id = item.get_group(0);
    auto local_id = item.get_local_linear_id();

    // allocate shared memory for expert index of current token
    sycl::
        multi_ptr<int[GroupWorkItem], sycl::access::address_space::local_space>
            expert_cumsum = sycl::ext::oneapi::group_local_memory_for_overwrite<
                int[GroupWorkItem],
                sycl::group<1>>(item.get_group());
    auto& expert_cumsum_ptr = *expert_cumsum;

    int expert_val = 0;
    if (local_id < n_experts)
      expert_val = rows_for_experts[local_id];

    if (n_experts <= SgSize)
      expert_val = sub_group_inclusive_scan<int, SgSize>(item, expert_val);
    else
      expert_val = group_inclusive_scan<int, NumSg, SgSize>(item, expert_val);

    if (local_id < n_experts)
      expert_cumsum_ptr[local_id] = expert_val;
    item.barrier(sycl::access::fence_space::local_space);

    int indices[TopK], expert_local_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      indices[i] = topk_indices[token_id * TopK + i];
      expert_local_offset[i] = offsets[token_id * TopK + i];
    }

    int expert_row_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      int expert_id = indices[i];
      int start_offset = expert_id == 0 ? 0 : expert_cumsum_ptr[expert_id - 1];
      expert_row_offset[i] = start_offset + expert_local_offset[i];
    }

    const T* activation_base =
        activations + token_id * n_channels + local_id * ElemsPerItem;
    T* reordered_activation_bases[TopK];
    for (int i = 0; i < TopK; ++i) {
      reordered_activation_bases[i] = reordered_activation +
          expert_row_offset[i] * n_channels + local_id * ElemsPerItem;
    }

    constexpr int stride = GroupWorkItem * ElemsPerItem;
    const int loop_count = (n_channels + stride - 1) / stride;

    for (int loop = 0; loop < loop_count; ++loop) {
      if (loop * stride + local_id * ElemsPerItem < n_channels) {
        T data[ElemsPerItem];
#pragma unroll
        for (int j = 0; j < ElemsPerItem; ++j) {
          data[j] = *(activation_base + loop * stride + j);
        }
#pragma unroll
        for (int i = 0; i < TopK; ++i) {
#pragma unroll
          for (int j = 0; j < ElemsPerItem; ++j) {
            *(reordered_activation_bases[i] + loop * stride + j) = data[j];
          }
        }
      }
    }
    if (local_id == 0) {
      for (int i = 0; i < TopK; ++i) {
        mapped_slot[token_id * TopK + i] = expert_row_offset[i];
      }
    }
  }

  const T* activations; // [n_tokens, n_channels]
  const int* rows_for_experts; // [n_experts]
  const int* topk_indices; // [n_tokens, num_top_k]
  const int* offsets; // [n_tokens, num_top_k]
  T* reordered_activation;
  int* mapped_slot; // [n_tokens, num_top_k]
  const int n_tokens;
  const int n_experts;
  const int n_channels;
};

template <int TopK, typename T>
void launch_moe_scatter(
    const T* activations,
    const int* rows_for_experts,
    const int* topk_indices,
    const int* offsets,
    T* reordered_activation,
    int* mapped_slot,
    const int n_tokens,
    const int n_experts,
    const int n_channels) {
  using Kernel = MoEScatter<TopK, T>;
  // TODO: maybe add template for GroupWorkItem in the future
  TORCH_CHECK(
      Kernel::GroupWorkItem >= n_experts,
      "MoEScatter::GroupWorkItem is expected to be larger than num_expert");
  TORCH_CHECK(
      n_channels % Kernel::ElemsPerItem == 0,
      "n_channels is expected to be aligned to Kernel::ElemsPerItem");

  auto range = Kernel::get_nd_range(n_tokens);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        activations,
        rows_for_experts,
        topk_indices,
        offsets,
        reordered_activation,
        mapped_slot,
        n_tokens,
        n_experts,
        n_channels);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T>
void moe_scatter(
    const T* activations,
    const int* rows_for_experts,
    const int* topk_indices,
    const int* offsets,
    T* reordered_activation,
    int* mapped_slot,
    const int n_tokens,
    const int n_experts,
    const int n_channels,
    const int n_topk) {
  switch (n_topk) {
    case 1:
      launch_moe_scatter<1, T>(
          activations,
          rows_for_experts,
          topk_indices,
          offsets,
          reordered_activation,
          mapped_slot,
          n_tokens,
          n_experts,
          n_channels);
      break;
    case 2:
      launch_moe_scatter<2, T>(
          activations,
          rows_for_experts,
          topk_indices,
          offsets,
          reordered_activation,
          mapped_slot,
          n_tokens,
          n_experts,
          n_channels);
      break;
    default:
      TORCH_CHECK(
          false,
          "error: not support topk=%d, up to topk=%d\n",
          n_topk,
          MaxTopK);
  };
};
}; // namespace MoEScatterImpl

namespace MoEGatherImpl {
// Re-gather the outputs of MoE and scale them by the gating score.
// Note: this kernel reset rows_for_experts to zero
template <int TopK, typename T>
struct MoEGather {
  static constexpr int GroupWorkItem = 256;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  static constexpr int Stride = ElemsPerItem * GroupWorkItem;
  MoEGather(
      T* layer_output,
      int32_t* rows_for_experts,
      const T* moe_output,
      const float* scores,
      const int32_t* mapped_slots,
      const int32_t n_channels,
      const int32_t n_experts,
      const int32_t n_tokens,
      const bool normalize_scales)
      : layer_output(layer_output),
        rows_for_experts(rows_for_experts),
        moe_output(moe_output),
        scores(scores),
        mapped_slots(mapped_slots),
        n_channels(n_channels),
        n_experts(n_experts),
        n_tokens(n_tokens),
        normalize_scales(normalize_scales) {}

  static inline sycl::nd_range<1> get_nd_range(const int n_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(n_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()(sycl::nd_item<1> item) const {
    auto token_idx = item.get_group(0);
    auto local_id = item.get_local_linear_id();
    int32_t token_mapped_slots[TopK];

    for (int i = 0; i < TopK; i++) {
      token_mapped_slots[i] = mapped_slots[token_idx * TopK + i];
    }

    if (token_idx == 0) {
      // Reset expert counts for its next use.
      if (local_id < n_experts) {
        rows_for_experts[local_id] = 0;
      }
    }

    float token_scores[TopK];
    for (int i = 0; i < TopK; i++) {
      token_scores[i] = scores[token_idx * TopK + i];
    }

    if (normalize_scales) {
      // Normalize the scores so that they sum to 1.
      float sum = 0.0f;
      for (int i = 0; i < TopK; i++) {
        sum += token_scores[i];
      }

      if (sum > 0.0f) {
        for (int i = 0; i < TopK; i++) {
          token_scores[i] /= sum;
        }
      }
    }

    const int32_t channel_offset = local_id * ElemsPerItem;

    const T* moe_output_bases[TopK];
#pragma unroll
    for (int i = 0; i < TopK; i++) {
      moe_output_bases[i] =
          moe_output + token_mapped_slots[i] * n_channels + channel_offset;
    }

    T* layer_output_base =
        layer_output + token_idx * n_channels + channel_offset;

    const int loop_count = (n_channels + Stride - 1) / Stride;
    for (int i = 0; i < loop_count; i++) {
      if (i * Stride + channel_offset < n_channels) {
        sycl::vec<float, ElemsPerItem> accum_buffer;
        for (int j = 0; j < ElemsPerItem; j++) {
          accum_buffer[j] = 0.0f;
        }

#pragma unroll
        for (int j = 0; j < TopK; j++) {
          sycl::vec<T, ElemsPerItem> reg_buffer;
          reg_buffer = *(reinterpret_cast<const sycl::vec<T, ElemsPerItem>*>(
              moe_output_bases[j] + i * Stride));

#pragma unroll
          for (int k = 0; k < ElemsPerItem; k++) {
            float up_cast = static_cast<float>(reg_buffer[k]);
            accum_buffer[k] += up_cast * token_scores[j];
          }
        }

        sycl::vec<T, ElemsPerItem> store_buffer;
#pragma unroll
        for (int j = 0; j < ElemsPerItem; j++) {
          store_buffer[j] = static_cast<T>(accum_buffer[j]);
        }
        *(reinterpret_cast<sycl::vec<T, ElemsPerItem>*>(
            layer_output_base + i * Stride)) = store_buffer;
      }
    }
  }

  T* layer_output; // [n_tokens, hidden_size]
  int32_t* rows_for_experts; // [n_experts]
  const T* moe_output; // [n_tokens * n_topk, hidden_size]
  const float* scores; // [n_tokens, n_topk]
  const int32_t* mapped_slots; // [n_tokens, n_topk]
  const int32_t n_channels;
  const int32_t n_experts;
  const int32_t n_tokens;
  const bool normalize_scales;
};

template <int TopK>
struct MoEGather<TopK, sycl::ext::oneapi::bfloat16> {
  using T = sycl::ext::oneapi::bfloat16;
  static constexpr int GroupWorkItem = 256;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  static constexpr int Stride = ElemsPerItem * GroupWorkItem;
  MoEGather(
      T* layer_output,
      int32_t* rows_for_experts,
      const T* moe_output,
      const float* scores,
      const int32_t* mapped_slots,
      const int32_t n_channels,
      const int32_t n_experts,
      const int32_t n_tokens,
      const bool normalize_scales)
      : layer_output(layer_output),
        rows_for_experts(rows_for_experts),
        moe_output(moe_output),
        scores(scores),
        mapped_slots(mapped_slots),
        n_channels(n_channels),
        n_experts(n_experts),
        n_tokens(n_tokens),
        normalize_scales(normalize_scales) {}

  static inline sycl::nd_range<1> get_nd_range(const int n_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(n_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()(sycl::nd_item<1> item) const {
    auto token_idx = item.get_group(0);
    auto local_id = item.get_local_linear_id();
    int32_t token_mapped_slots[TopK];

    for (int i = 0; i < TopK; i++) {
      token_mapped_slots[i] = mapped_slots[token_idx * TopK + i];
    }

    if (token_idx == 0) {
      // Reset expert counts for its next use.
      if (local_id < n_experts) {
        rows_for_experts[local_id] = 0;
      }
    }

    float token_scores[TopK];
    for (int i = 0; i < TopK; i++) {
      token_scores[i] = scores[token_idx * TopK + i];
    }

    if (normalize_scales) {
      // Normalize the scores so that they sum to 1.
      float sum = 0.0f;
      for (int i = 0; i < TopK; i++) {
        sum += token_scores[i];
      }

      if (sum > 0.0f) {
        for (int i = 0; i < TopK; i++) {
          token_scores[i] /= sum;
        }
      }
    }

    const int32_t channel_offset = local_id * ElemsPerItem;

    const T* moe_output_bases[TopK];
#pragma unroll
    for (int i = 0; i < TopK; i++) {
      moe_output_bases[i] =
          moe_output + token_mapped_slots[i] * n_channels + channel_offset;
    }

    T* layer_output_base =
        layer_output + token_idx * n_channels + channel_offset;

    const int loop_count = (n_channels + Stride - 1) / Stride;
    for (int i = 0; i < loop_count; i++) {
      if (i * Stride + channel_offset < n_channels) {
        sycl::vec<float, ElemsPerItem> accum_buffer;
        for (int j = 0; j < ElemsPerItem; j++) {
          accum_buffer[j] = 0.0f;
        }

#pragma unroll
        for (int j = 0; j < TopK; j++) {
          T reg_buffer[ElemsPerItem];
#pragma unroll
          for (int k = 0; k < ElemsPerItem; ++k) {
            reg_buffer[k] = *(moe_output_bases[j] + i * Stride + k);
          }

#pragma unroll
          for (int k = 0; k < ElemsPerItem; k++) {
            float up_cast = static_cast<float>(reg_buffer[k]);
            accum_buffer[k] += up_cast * token_scores[j];
          }
        }

        T store_buffer[ElemsPerItem];
        for (int j = 0; j < ElemsPerItem; j++) {
          store_buffer[j] = static_cast<T>(accum_buffer[j]);
        }
#pragma unroll
        for (int j = 0; j < ElemsPerItem; j++) {
          layer_output_base[i * Stride + j] = store_buffer[j];
        }
      }
    }
  }

  T* layer_output; // [n_tokens, hidden_size]
  int32_t* rows_for_experts; // [n_experts]
  const T* moe_output; // [n_tokens * n_topk, hidden_size]
  const float* scores; // [n_tokens, n_topk]
  const int32_t* mapped_slots; // [n_tokens, n_topk]
  const int32_t n_channels;
  const int32_t n_experts;
  const int32_t n_tokens;
  const bool normalize_scales;
};

template <int TopK, typename T>
void launch_moe_gather(
    T* layer_output,
    const T* moe_output,
    const float* scores,
    const int32_t* mapped_slots,
    int32_t* rows_for_experts,
    const int32_t n_channels,
    const int32_t n_experts,
    const int32_t n_tokens,
    const bool normalize_scales) {
  using Kernel = MoEGather<TopK, T>;
  // TODO: maybe add template for GroupWorkItem in the future
  TORCH_CHECK(
      Kernel::GroupWorkItem >= n_experts,
      "MoEScatter::GroupWorkItem is expected to be larger than num_expert");
  TORCH_CHECK(
      n_channels % Kernel::ElemsPerItem == 0,
      "n_channels is expected to be aligned to Kernel::ElemsPerItem");
  auto range = Kernel::get_nd_range(n_tokens);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        layer_output,
        rows_for_experts,
        moe_output,
        scores,
        mapped_slots,
        n_channels,
        n_experts,
        n_tokens,
        normalize_scales);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

//
template <typename T>
void moe_gather(
    T* layer_output,
    const T* moe_output,
    const float* scores,
    const int32_t* mapped_slots,
    int32_t* rows_for_experts,
    const int32_t n_channels,
    const int32_t n_experts,
    const int32_t n_tokens,
    const int32_t n_top_k,
    const bool normalize_scales) {
  switch (n_top_k) {
    case 1:
      launch_moe_gather<1, T>(
          layer_output,
          moe_output,
          scores,
          mapped_slots,
          rows_for_experts,
          n_channels,
          n_experts,
          n_tokens,
          normalize_scales);
      break;
    case 2:
      launch_moe_gather<2, T>(
          layer_output,
          moe_output,
          scores,
          mapped_slots,
          rows_for_experts,
          n_channels,
          n_experts,
          n_tokens,
          normalize_scales);
      break;
    default:
      TORCH_CHECK(
          false,
          "error in moe_gather kernel: not support topk=%d, up to topk=%d",
          n_top_k,
          MaxTopK);
  };
};

} // namespace MoEGatherImpl

namespace MoESumImpl {
template <typename scalar_t, int TopK>
struct MoESum {
  MoESum(scalar_t* output, const scalar_t* input, int d)
      : output(output), input(input), d(d) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t group_id = item.get_group(0);
    const int64_t group_size = item.get_local_range(0);
    const int64_t local_idx = item.get_local_id(0);
    const int64_t linear_idx = item.get_global_linear_id();

    for (int64_t idx = local_idx; idx < d; idx += group_size) {
      scalar_t x = 0.0;
#pragma unroll
      for (int64_t k = 0; k < TopK; k++) {
        x += input[group_id * TopK * d + k * d + idx]; // block offset + top_k
                                                       // offset + local_id
      }
      output[linear_idx] = x;
    }
  }

  scalar_t* output; // [..., d]
  const scalar_t* input; // [..., topk, d]
  int d;
};

template <typename T, int TopK>
void launch_moe_sum(
    T* output,
    const T* input,
    int hidden_size,
    int num_tokens) {
  using Kernel = MoESum<T, TopK>;
  int wg_size = std::min(hidden_size, 1024);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(output, input, hidden_size);

    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range(num_tokens), sycl::range(wg_size)), task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T>
void moe_sum(
    at::Tensor& input, // [num_tokens, topk, hidden_size]
    at::Tensor& output) // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const int global_size = output.numel();
  const int topk = input.size(1);

  switch (topk) {
    case 2:
      launch_moe_sum<T, 2>(
          output.data_ptr<T>(), input.data_ptr<T>(), hidden_size, global_size);
      break;

    case 3:
      launch_moe_sum<T, 3>(
          output.data_ptr<T>(), input.data_ptr<T>(), hidden_size, global_size);
      break;

    case 4:
      launch_moe_sum<T, 4>(
          output.data_ptr<T>(), input.data_ptr<T>(), hidden_size, global_size);
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

} // namespace MoESumImpl

/**
 * @brief Perform topk after softmax on gating_output.
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param n_topk The number of top experts to select.
 * @return A tuple of tensors (topk_weights, topk_indices, rows_for_experts,
 * offsets).
 */
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> topk_softmax(
    const Tensor& gating_output,
    const int64_t n_topk) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  int n_tokens = shape[0];
  int n_experts = shape[1];

  auto topk_weights =
      at::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  auto rows_for_experts =
      at::zeros({n_experts}, at::dtype(at::kInt).device(at::kXPU));
  auto offsets =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      gating_output.scalar_type(),
      "fused_topk_softmax_kernel",
      [&]() {
        TopKSoftmaxImpl::fused_topk_softmax<scalar_t>(
            reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
            reinterpret_cast<float*>(topk_weights.data_ptr()),
            reinterpret_cast<int*>(topk_indices.data_ptr()),
            reinterpret_cast<int*>(rows_for_experts.data_ptr()),
            reinterpret_cast<int*>(offsets.data_ptr()),
            n_tokens,
            n_experts,
            n_topk);
      });

  return std::make_tuple(topk_weights, topk_indices, rows_for_experts, offsets);
}

/**
 * @brief Perform grouped topk after sigmoid/addbias on gating_output.
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param n_topk The number of top experts to select.
 * @param n_topk_group The number of top experts to select in the group.
 * @return A tuple of tensors (topk_weights, topk_indices, rows_for_experts,
 * offsets).
 */
static std::tuple<at::Tensor, at::Tensor> grouped_topk_sigmoid(
    const Tensor& hidden_states,
    const Tensor& gating_output,
    const int64_t n_topk,
    const bool renormalize,
    const int64_t n_expert_group,
    const int64_t n_topk_group,
    const c10::string_view scoring_func,
    const Tensor& bias) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      hidden_states.sizes()[0] == gating_output.sizes()[0],
      "Number of tokens mismatch")
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  auto shape_bias = bias.sizes().vec();
  TORCH_CHECK(
      shape_bias[0] == shape[1],
      "gating_output and bias must has same innermost dimension, but got ",
      shape,
      " and ",
      shape_bias);
  int n_tokens = shape[0];
  int n_experts = shape[1];

  auto topk_weights =
      at::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  auto score_buf = at::empty_like(gating_output, at::dtype(at::kFloat));
  auto max_buf = at::empty(
      {n_tokens, n_expert_group}, at::dtype(at::kFloat).device(at::kXPU));
  if (gating_output.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    GroupedTopKImpl::fused_grouped_topk_sigmoid<scalar_t>(
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        reinterpret_cast<scalar_t*>(bias.data_ptr()),
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<float*>(score_buf.data_ptr()),
        reinterpret_cast<float*>(max_buf.data_ptr()),
        n_tokens,
        n_experts,
        n_expert_group,
        n_topk,
        n_topk_group);
  } else if (gating_output.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    GroupedTopKImpl::fused_grouped_topk_sigmoid<scalar_t>(
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        reinterpret_cast<scalar_t*>(bias.data_ptr()),
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<float*>(score_buf.data_ptr()),
        reinterpret_cast<float*>(max_buf.data_ptr()),
        n_tokens,
        n_experts,
        n_expert_group,
        n_topk,
        n_topk_group);
  } else {
    using scalar_t = float;
    GroupedTopKImpl::fused_grouped_topk_sigmoid<scalar_t>(
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        reinterpret_cast<scalar_t*>(bias.data_ptr()),
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<float*>(score_buf.data_ptr()),
        reinterpret_cast<float*>(max_buf.data_ptr()),
        n_tokens,
        n_experts,
        n_expert_group,
        n_topk,
        n_topk_group);
  }
  return std::make_tuple(topk_weights, topk_indices);
}

/**
 * @brief moe_scatter is a reordering kernel for moe that reorders the
 * activations based on the topk_indices and offsets.
 *
 * @param activations Original activations tensor of shape [n_tokens,
 * n_channels].
 * @param rows_for_experts The number of tokens assigned to each expert.
 * @param topk_indices The indices of the topk experts for each token.
 * @param expert_offsets The offsets of the topk experts for each token.
 * @param n_experts The number of experts.
 * @param n_topk The number of top experts to select.
 * @return A tuple of tensors (reordered_activation, mapped_slot).
 */

static std::tuple<at::Tensor, at::Tensor> moe_scatter(
    const Tensor& activation, // [n_tokens, n_channels]
    const Tensor& rows_for_experts, // [n_experts]
    const Tensor& topk_indices, //[n_tokens, n_topk]
    const Tensor& expert_offsets, //[n_tokens, n_topk]
    const int64_t n_experts,
    const int64_t n_topk) {
  auto shape = activation.sizes().vec();
  TORCH_CHECK(
      shape.size() == 2,
      "activation must be 2D tensor, but got ",
      shape.size(),
      "D");
  int n_tokens = shape[0];
  int n_channels = shape[1];

  auto reordered_activation =
      at::empty({n_tokens * n_topk, n_channels}, activation.options());
  auto mapped_slot =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));

  if (activation.scalar_type() == at::kHalf) {
    MoEScatterImpl::moe_scatter<sycl::half>(
        reinterpret_cast<sycl::half*>(activation.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<int*>(expert_offsets.data_ptr()),
        reinterpret_cast<sycl::half*>(reordered_activation.data_ptr()),
        reinterpret_cast<int*>(mapped_slot.data_ptr()),
        n_tokens,
        n_experts,
        n_channels,
        n_topk);
  } else if (activation.scalar_type() == at::kBFloat16) {
    MoEScatterImpl::moe_scatter<sycl::ext::oneapi::bfloat16>(
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(activation.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<int*>(expert_offsets.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
            reordered_activation.data_ptr()),
        reinterpret_cast<int*>(mapped_slot.data_ptr()),
        n_tokens,
        n_experts,
        n_channels,
        n_topk);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(
        activation.scalar_type(), "moe_scatter", [&]() {
          MoEScatterImpl::moe_scatter<scalar_t>(
              reinterpret_cast<scalar_t*>(activation.data_ptr()),
              reinterpret_cast<int*>(rows_for_experts.data_ptr()),
              reinterpret_cast<int*>(topk_indices.data_ptr()),
              reinterpret_cast<int*>(expert_offsets.data_ptr()),
              reinterpret_cast<scalar_t*>(reordered_activation.data_ptr()),
              reinterpret_cast<int*>(mapped_slot.data_ptr()),
              n_tokens,
              n_experts,
              n_channels,
              n_topk);
        });
  }

  return std::make_tuple(reordered_activation, mapped_slot);
}

/**
 * @brief moe_gather is a gather kernel for moe that gathers the activations
 * based on the mapped_slot and scores.
 *
 * @param moe_output The output tensor of shape [n_tokens * n_topk, n_channels]
 * that has been reordered by moe_scatter.
 * @param scores The gating scores tensor of shape [n_tokens, n_topk].
 * @param mapped_slot The mapped slot tensor of shape [n_tokens, n_topk].
 * @param rows_for_experts The number of tokens assigned to each expert.
 * @param n_experts The number of experts.
 * @param n_topk The number of top experts to select.
 * @param normalize_scales Whether to normalize the gating scores.
 * @return The gathered output tensor of shape [n_tokens, n_channels].
 */
static at::Tensor moe_gather(
    const Tensor& moe_output, // [n_tokens * n_topk, n_channels]
    const Tensor& scores, // [n_tokens, n_topk]
    const Tensor& mapped_slot, //[n_tokens, n_topk]
    Tensor& rows_for_experts, //[n_experts]
    const int64_t n_experts,
    const int64_t n_topk,
    const bool normalize_scales) {
  int n_tokens = scores.sizes()[0];
  int n_channels = moe_output.sizes()[1];

  auto gathered_output =
      at::empty({n_tokens, n_channels}, moe_output.options());
  if (moe_output.scalar_type() == at::kHalf) {
    MoEGatherImpl::moe_gather<sycl::half>(
        reinterpret_cast<sycl::half*>(gathered_output.data_ptr()),
        reinterpret_cast<sycl::half*>(moe_output.data_ptr()),
        reinterpret_cast<float*>(scores.data_ptr()),
        reinterpret_cast<int*>(mapped_slot.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        n_channels,
        n_experts,
        n_tokens,
        n_topk,
        normalize_scales);
  } else if (moe_output.scalar_type() == at::kBFloat16) {
    MoEGatherImpl::moe_gather<sycl::ext::oneapi::bfloat16>(
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
            gathered_output.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(moe_output.data_ptr()),
        reinterpret_cast<float*>(scores.data_ptr()),
        reinterpret_cast<int*>(mapped_slot.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        n_channels,
        n_experts,
        n_tokens,
        n_topk,
        normalize_scales);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(moe_output.scalar_type(), "moe_gather", [&]() {
      MoEGatherImpl::moe_gather<scalar_t>(
          reinterpret_cast<scalar_t*>(gathered_output.data_ptr()),
          reinterpret_cast<scalar_t*>(moe_output.data_ptr()),
          reinterpret_cast<float*>(scores.data_ptr()),
          reinterpret_cast<int*>(mapped_slot.data_ptr()),
          reinterpret_cast<int*>(rows_for_experts.data_ptr()),
          n_channels,
          n_experts,
          n_tokens,
          n_topk,
          normalize_scales);
    });
  }
  return gathered_output;
}

/**
 * @brief moe_sum is a sum kernel for moe that sum all the topK hidden sizes
 * into 1 hidden size
 *
 * @param input The input tensor of shape [num_tokens, topk, hidden_size]
 * that has been generated by the moe
 * @param output The compressed hidden size tensor of shape [num_tokens,
 * hidden_size].
 */
static void moe_sum(
    at::Tensor& input, // [num_tokens, topk, hidden_size]
    at::Tensor& output) { // [num_tokens, hidden_size]

  const int hidden_size = input.size(-1);

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, output.scalar_type(), "moe_sum", [&]() {
        MoESumImpl::moe_sum<scalar_t>(input, output);
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("topk_softmax.moe", at::AtenIpexTypeXPU::topk_softmax);
  IPEX_OP_REGISTER(
      "grouped_topk_sigmoid.moe", at::AtenIpexTypeXPU::grouped_topk_sigmoid);
  IPEX_OP_REGISTER("moe_scatter.moe", at::AtenIpexTypeXPU::moe_scatter);
  IPEX_OP_REGISTER("moe_gather.moe", at::AtenIpexTypeXPU::moe_gather);
  IPEX_OP_REGISTER("moe_sum.moe", at::AtenIpexTypeXPU::moe_sum)
}
} // namespace
