#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <sycl/sycl.hpp>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "utils/ComputeEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace TopKSoftmaxImpl {
// Each WI compute  one token
template <typename T>
struct FusedTopkSoftmax {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = 8;
  static constexpr float kNegInfinity = INFINITY * -1;

  FusedTopkSoftmax(
      float* topk_weights,
      int* topk_ids,
      int* rows_for_experts,
      int* offsets,
      const T* gating_output,
      const bool renormalize,
      const int tokens,
      const int experts,
      const int top_k)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        rows_for_experts(rows_for_experts),
        offsets(offsets),
        gating_output(gating_output),
        renormalize(renormalize),
        tokens(tokens),
        experts(experts),
        top_k(top_k) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int tokens,
      const int experts) {
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;
    int group_size = (experts + calc_per_item - 1) / calc_per_item;
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group =
        (group_size + sub_group_size - 1) / sub_group_size;
    group_size = sub_groups_per_group * sub_group_size;
    int global_size =
        (tokens + sub_groups_per_group - 1) / sub_groups_per_group;

    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(1, 1, global_size);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline T Sigmoid(T x) {
    float sycl_x = static_cast<float>(x);
    float result = 1.0f / (1.0f + sycl::exp(-sycl_x));
    return static_cast<T>(result);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(
      sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;

    if (tid >= tokens) {
      return; // Out of bounds
    }

    T local_elems[malloc_per_item];
    int local_idx[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;

    if (start_offset + local_num >= experts) {
      local_num = experts - start_offset;
      if (local_num < 0) {
        local_num = 0; // No elements to process
      }
    }

    for (int e = 0; e < calc_per_item; ++e) {
      local_elems[e] = kNegInfinity;
      local_idx[e] = -1;
    }

    for (int e = 0; e < local_num; ++e) {
      local_elems[e] = gating_output[tid * experts + start_offset + e];
      local_idx[e] = start_offset + e;
    }

    // Perform top-k selection
    T topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    for (int k = 0; k < top_k; ++k) {
      T k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;
      for (int e = 0; e < calc_per_item; ++e) {
        T my_val = local_elems[e];
        int my_idx = local_idx[e];
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          T other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val ||
              (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }
        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;

          if (k_max_idx == local_idx[e]) {
            remove_ix = e; // Mark this index for removal
          } else
            remove_ix = -1;
        }
      }
      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx;
      if (remove_ix != -1) {
        // Reset the score to avoid re-selection
        local_elems[remove_ix] = kNegInfinity;
        local_idx[remove_ix] = -1;
        remove_ix = -1;
      }
    }

    float max_score = topk_weights_local[0];
    float sum_exp = 0;

    for (int i = 0; i < top_k; ++i) {
      float score = topk_weights_local[i];
      sum_exp += sycl::exp(score - max_score);
    }

    for (int e = 0; e < calc_per_item; ++e) {
      float score = local_elems[e];
      float my_val = sycl::exp(score - max_score);
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
        my_val += other_val;
      }
      sum_exp += my_val;
    }

    for (int i = 0; i < top_k; ++i) {
      float score = topk_weights_local[i];
      topk_weights_local[i] = sycl::exp(score - max_score) / sum_exp;
    }

    if (renormalize) {
      // Renormalize the top-k weights
      float sum = 0;
      for (int i = 0; i < top_k; ++i) {
        sum += topk_weights_local[i];
      }
      if (sum > 0) {
        for (int i = 0; i < top_k; ++i) {
          topk_weights_local[i] /= sum;
        }
      }
    }

    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < top_k; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        if (topk_ids_local[i] < 0 || topk_ids_local[i] >= experts) {
          // Ensure valid index
          topk_ids[offset + i] = 0;
          offsets[offset + i] = 0;
          continue;
        }
        topk_ids[offset + i] = topk_ids_local[i];
        auto ref_num_tokens = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>(
            *(rows_for_experts + topk_ids_local[i]));
        int old = ref_num_tokens.fetch_add(1);
        offsets[offset + i] = old;
      }
    }
  }
  float* topk_weights;
  int* topk_ids;
  int* rows_for_experts;
  int* offsets;
  const T* gating_output;
  const bool renormalize;
  const int tokens;
  const int experts;
  const int top_k;
};

template <typename T>
void launch_fused_topk_softmax(
    sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    int* rows_for_experts,
    int* offsets,
    const bool renormalize,
    const int top_k,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSoftmax<T>;
  auto range = Kernel::get_nd_range(num_tokens, num_experts);

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        topk_weights,
        topk_indices,
        rows_for_experts,
        offsets,
        gating_output,
        renormalize,
        num_tokens,
        num_experts,
        top_k);
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
    const bool renormalize,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  auto& queue = dpcppGetCurrentQueue();

  launch_fused_topk_softmax(
      queue,
      gating_output,
      topk_weights,
      topk_indices,
      rows_for_experts,
      offsets,
      renormalize,
      topk,
      num_tokens,
      num_experts);
};
}; // namespace TopKSoftmaxImpl

namespace GroupedTopKImpl {

enum class ScoringFunc {
  DEFAULT = 0,
  SOFTMAX = 1,
  SIGMOID = 2,
};

template <typename T, int MAX_EXPERT_GROUPS>
struct Fused_Grouped_Topk {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = MAX_EXPERT_GROUPS;
  static constexpr float kNegInfinity = INFINITY * -1;

  Fused_Grouped_Topk(
      float* topk_weights,
      int* topk_ids,
      int* rows_for_experts,
      int* offsets,
      const T* gating_output,
      const T* e_score_correction_bias,
      const ScoringFunc scoring_mode,
      const bool renormalize,
      const int tokens,
      const int experts,
      const int top_k,
      const int num_expert_group,
      const int topk_group)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        rows_for_experts(rows_for_experts),
        offsets(offsets),
        gating_output(gating_output),
        e_score_correction_bias(e_score_correction_bias),
        scoring_mode(scoring_mode),
        renormalize(renormalize),
        tokens(tokens),
        experts(experts),
        top_k(top_k),
        num_expert_group(num_expert_group),
        topk_group(topk_group) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int tokens,
      const int experts) {
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;
    int group_size = (experts + calc_per_item - 1) / calc_per_item;
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group =
        (group_size + sub_group_size - 1) / sub_group_size;
    group_size = sub_groups_per_group * sub_group_size;
    int global_size =
        (tokens + sub_groups_per_group - 1) / sub_groups_per_group;

    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(1, 1, global_size);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float Sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(
      sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    int experts_per_group = experts / num_expert_group;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;

    if (tid >= tokens) {
      return; // Out of bounds
    }

    T load_elems[malloc_per_item];
    int local_idx[malloc_per_item];
    T bias[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;

    if (start_offset + local_num >= experts) {
      local_num = experts - start_offset;
      if (local_num < 0) {
        local_num = 0; // No elements to process
      }
    }

    for (int e = 0; e < calc_per_item; ++e) {
      load_elems[e] = kNegInfinity;
      local_idx[e] = -1;
      bias[e] = 0.0f; // Initialize bias to zero
    }

    for (int e = 0; e < local_num; ++e) {
      load_elems[e] = gating_output[tid * experts + start_offset + e];
    }

    float local_elems[malloc_per_item];

    for (int e = 0; e < local_num; ++e) {
      local_elems[e] = load_elems[e];
      local_idx[e] = start_offset + e;
    }

    if (scoring_mode == ScoringFunc::SOFTMAX) {
      float softmax_max = kNegInfinity;
      for (int e = 0; e < local_num; ++e) {
        softmax_max =
            (softmax_max > local_elems[e]) ? softmax_max : local_elems[e];
      }
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, softmax_max, offset);
        softmax_max = (softmax_max > other_val) ? softmax_max : other_val;
      }
      float softmax_sum = 0.0f;
      for (int e = 0; e < local_num; ++e) {
        float s = local_elems[e];
        softmax_sum += sycl::native::exp(s - softmax_max);
      }
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, softmax_sum, offset);
        softmax_sum += other_val;
      }
      for (int e = 0; e < local_num; ++e) {
        float s = local_elems[e];
        local_elems[e] = sycl::native::exp(s - softmax_max) / softmax_sum;
      }
    } else if (scoring_mode == ScoringFunc::SIGMOID) {
      for (int e = 0; e < local_num; ++e) {
        float s = load_elems[e];
        load_elems[e] = Sigmoid(s);
      }
      for (int e = 0; e < local_num; ++e) {
        local_elems[e] = load_elems[e];
      }
    }

    bool has_bias = e_score_correction_bias != nullptr;
    if (has_bias) {
      for (int e = 0; e < local_num; ++e) {
        bias[e] = e_score_correction_bias[start_offset + e];
      }
    }

    // perform topk_group groups
    // 1 calculate each group scores
    float group_scores[malloc_per_item * 2];
    for (int i = 0; i < num_expert_group * 2; ++i) {
      group_scores[i] = kNegInfinity;
    }
    for (int i = 0; i < local_num; ++i) {
      float b = bias[i];
      float score = local_elems[i] + b;
      int i_group = (calc_per_item * sg_local_id + i) / experts_per_group;
      float group_max = group_scores[i_group];
      float group_next_max = group_scores[num_expert_group + i_group];
      if (score > group_max) {
        group_next_max = group_max;
        group_max = score;
      } else if (score > group_next_max) {
        group_next_max = score;
      }
      group_scores[i_group] = group_max;
      group_scores[num_expert_group + i_group] = group_next_max;
    }
    for (int i = 0; i < num_expert_group; ++i) {
      float group_max = group_scores[i];
      float group_next_max = group_scores[num_expert_group + i];

      float max1 = sycl::reduce_over_group(
          sg, sycl::max(group_max, group_next_max), sycl::maximum<>());
      float local_second =
          (group_max < max1 && group_max > -INFINITY) ? group_max : -INFINITY;
      local_second = (group_next_max < max1 && group_next_max > local_second)
          ? group_next_max
          : local_second;
      float max2 = sycl::reduce_over_group(sg, local_second, sycl::maximum<>());
      group_scores[i] = max1 + (has_bias ? max2 : 0.0f);
    }

    // 2 find topk_group groups as kNegInfinity
    int group_topk_idx[malloc_per_item];
    for (int k = 0; k < topk_group; ++k) {
      float k_max = group_scores[0];
      int k_max_idx = 0;
      for (int e = 1; e < num_expert_group; ++e) {
        float score = group_scores[e];

        if (score > k_max) {
          k_max = score;
          k_max_idx = e;
        }
      }
      group_scores[k_max_idx] = kNegInfinity;
      group_topk_idx[k] = k_max_idx;
    }

    // 3 mask no-topk_group groups
    for (int i = 0; i < calc_per_item; ++i) {
      bool is_masked = true;
      for (int k = 0; k < topk_group; ++k) {
        if ((local_idx[i] / experts_per_group) == group_topk_idx[k]) {
          is_masked = false;
          break;
        }
      }
      if (is_masked) {
        local_elems[i] = kNegInfinity;
      }
    }

    // Perform top-k selection
    float topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    for (int k = 0; k < top_k; ++k) {
      float k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;
      for (int e = 0; e < calc_per_item; ++e) {
        float le = local_elems[e];
        float b = bias[e];
        float my_val = le + b;
        int my_idx = local_idx[e];
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val ||
              (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }
        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;

          if (k_max_idx == local_idx[e]) {
            remove_ix = e; // Mark this index for removal
          } else
            remove_ix = -1;
        }
      }

      int select_item = k_max_idx / calc_per_item;
      int select_elem = k_max_idx % calc_per_item;
      k_max = local_elems[select_elem];
      k_max = sycl::group_broadcast(sg, k_max, select_item);
      if (remove_ix != -1) {
        local_elems[remove_ix] =
            kNegInfinity; // Reset the score to avoid re-selection
        local_idx[remove_ix] = -1;
        remove_ix = -1;
      }

      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx < 0 ? k : k_max_idx;
    }

    if (renormalize) {
      // Renormalize the top-k weights
      float sum = 0;
      for (int i = 0; i < top_k; ++i) {
        sum += topk_weights_local[i];
      }
      if (sum > 0) {
        for (int i = 0; i < top_k; ++i) {
          topk_weights_local[i] /= sum;
        }
      }
    }

    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < top_k; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        if (!(topk_ids_local[i] >= 0 && topk_ids_local[i] < experts)) {
          // Ensure valid index
          topk_ids[offset + i] = 0;
          offsets[offset + i] = 0;
          continue;
        }
        topk_ids[offset + i] = topk_ids_local[i];
        auto ref_num_tokens = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>(
            *(rows_for_experts + topk_ids_local[i]));
        int old = ref_num_tokens.fetch_add(1);
        offsets[offset + i] = old;
      }
    }
  }
  float* topk_weights;
  int* topk_ids;
  int* rows_for_experts;
  int* offsets;
  const T* gating_output;
  const T* e_score_correction_bias;
  const ScoringFunc scoring_mode;
  const bool renormalize;
  const int tokens;
  const int experts;
  const int top_k;
  const int num_expert_group;
  const int topk_group;
};

template <typename T, int MAX_EXPERT_GROUPS>
void launch_fused_grouped_topk(
    sycl::queue& queue,
    float* topk_weights,
    int* topk_ids,
    int* rows_for_experts,
    int* offsets,
    const T* gating_output,
    const T* e_score_correction_bias,
    const ScoringFunc scoring_mode,
    const bool renormalize,
    const int tokens,
    const int experts,
    const int top_k,
    const int num_expert_group,
    const int topk_group) {
  using Kernel = Fused_Grouped_Topk<T, MAX_EXPERT_GROUPS>;
  auto range = Kernel::get_nd_range(tokens, experts);

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        topk_weights,
        topk_ids,
        rows_for_experts,
        offsets,
        gating_output,
        e_score_correction_bias,
        scoring_mode,
        renormalize,
        tokens,
        experts,
        top_k,
        num_expert_group,
        topk_group);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T>
void fused_grouped_topk(
    float* topk_weights,
    int* topk_ids,
    int* rows_for_experts,
    int* offsets,
    const T* gating_output,
    const T* e_score_correction_bias,
    const ScoringFunc scoring_mode,
    const bool renormalize,
    const int tokens,
    const int experts,
    const int top_k,
    const int num_expert_group,
    const int topk_group) {
  auto& queue = dpcppGetCurrentQueue();

  TORCH_CHECK(
      topk_group <= num_expert_group,
      "topk_group must be less than or equal to num_expert_group");
  TORCH_CHECK(
      experts % num_expert_group == 0,
      "The number of experts (experts=",
      experts,
      ") must be divisible by num_expert_group (",
      num_expert_group,
      ").");

  int max_expert_group = ((num_expert_group + 7) / 8) * 8;
#define CASE_TOPK(K)                 \
  case K:                            \
    launch_fused_grouped_topk<T, K>( \
        queue,                       \
        topk_weights,                \
        topk_ids,                    \
        rows_for_experts,            \
        offsets,                     \
        gating_output,               \
        e_score_correction_bias,     \
        scoring_mode,                \
        renormalize,                 \
        tokens,                      \
        experts,                     \
        top_k,                       \
        num_expert_group,            \
        topk_group);                 \
    break;
  switch (max_expert_group) {
    CASE_TOPK(8)
    CASE_TOPK(16)
    default:
      TORCH_CHECK(
          false, "error: not support num_expert_group=%d,\n", num_expert_group);
  }
#undef CASE_TOPK
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
#define CASE_TOPK(K)          \
  case K:                     \
    launch_moe_scatter<K, T>( \
        activations,          \
        rows_for_experts,     \
        topk_indices,         \
        offsets,              \
        reordered_activation, \
        mapped_slot,          \
        n_tokens,             \
        n_experts,            \
        n_channels);          \
    break;
  switch (n_topk) {
    CASE_TOPK(1)
    CASE_TOPK(2)
    CASE_TOPK(4)
    CASE_TOPK(6)
    CASE_TOPK(8)
    default:
      TORCH_CHECK(false, "error: not support topk=%d,\n", topk);
  }
#undef CASE_TOPK
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
#define CASE_TOPK(K)         \
  case K:                    \
    launch_moe_gather<K, T>( \
        layer_output,        \
        moe_output,          \
        scores,              \
        mapped_slots,        \
        rows_for_experts,    \
        n_channels,          \
        n_experts,           \
        n_tokens,            \
        normalize_scales);   \
    break;
  switch (n_top_k) {
    CASE_TOPK(1)
    CASE_TOPK(2)
    CASE_TOPK(4)
    CASE_TOPK(6)
    CASE_TOPK(8)
    default:
      TORCH_CHECK(false, "error: not support topk=%d,\n", topk);
  }
#undef CASE_TOPK
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
    const int64_t n_topk,
    const bool renormalize) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  int n_tokens = shape[0];
  int n_experts = shape[1];

  TORCH_CHECK(
      n_experts <= 128,
      "n_experts only support up to 128, but got ",
      n_experts);

  int n_experts_aligned = (n_experts + 7) / 8 * 8; // align to 8
  auto topk_weights =
      at::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  auto rows_for_experts =
      at::zeros({n_experts_aligned}, at::dtype(at::kInt).device(at::kXPU));
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
            renormalize,
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
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> grouped_topk(
    const Tensor& hidden_states,
    const Tensor& gating_output,
    const int64_t n_topk,
    const bool renormalize,
    const int64_t n_expert_group,
    const int64_t n_topk_group,
    const c10::string_view scoring_func,
    const c10::optional<at::Tensor>& bias) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      hidden_states.sizes()[0] == gating_output.sizes()[0],
      "Number of tokens mismatch")
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  if (bias.has_value()) {
    auto shape_bias = bias->sizes().vec();
    TORCH_CHECK(
        shape_bias[0] == shape[1],
        "gating_output and bias must has same innermost dimension, but got ",
        shape,
        " and ",
        shape_bias);
  }
  int n_tokens = shape[0];
  int n_experts = shape[1];

  GroupedTopKImpl::ScoringFunc scoring_mode;
  if (scoring_func == "sigmoid") {
    scoring_mode = GroupedTopKImpl::ScoringFunc::SIGMOID;
  } else if (scoring_func == "softmax") {
    scoring_mode = GroupedTopKImpl::ScoringFunc::SOFTMAX;
  } else {
    scoring_mode = GroupedTopKImpl::ScoringFunc::DEFAULT;
  }

  int n_experts_aligned = (n_experts + 7) / 8 * 8; // align to 8

  auto topk_weights =
      at::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));
  auto rows_for_experts =
      at::zeros({n_experts_aligned}, at::dtype(at::kInt).device(at::kXPU));
  auto offsets =
      at::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));

  if (gating_output.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(offsets.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  } else if (gating_output.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(offsets.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  } else {
    using scalar_t = float;
    GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(offsets.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  }
  return std::make_tuple(topk_weights, topk_indices, rows_for_experts, offsets);
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
  IPEX_OP_REGISTER("grouped_topk.moe", at::AtenIpexTypeXPU::grouped_topk);
  IPEX_OP_REGISTER("moe_scatter.moe", at::AtenIpexTypeXPU::moe_scatter);
  IPEX_OP_REGISTER("moe_gather.moe", at::AtenIpexTypeXPU::moe_gather);
  IPEX_OP_REGISTER("moe_sum.moe", at::AtenIpexTypeXPU::moe_sum)
}
} // namespace
