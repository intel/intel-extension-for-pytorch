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
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static constexpr int WARP_SIZE = 32;
// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <int TPB, typename InputdType>
class moeSoftmax {
 public:
  moeSoftmax(
      sycl::local_accessor<float, 1>& slm,
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_cols)
      : slm(slm),
        input(input),
        finished(finished),
        output(output),
        num_cols(num_cols) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    void* slm_ptr = static_cast<void*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    float* normalizing_factor = reinterpret_cast<float*>(slm_ptr);
    float* float_max = normalizing_factor + 1;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int thread_row_offset = group_id_x * num_cols;

    float threadData(INFINITY * -1);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[group_id_x]) {
      return;
    }

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData = MAX(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem =
        sycl::reduce_over_group(group, threadData, sycl::maximum<float>());
    if (local_id_x == 0) {
      *float_max = maxElem;
    }
    item.barrier(sycl::access::fence_space::local_space);

    threadData = 0;

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData += sycl::exp((static_cast<float>(input[idx]) - *float_max));
    }

    const auto Z = sycl::reduce_over_group(group, threadData, sycl::plus<>());

    if (local_id_x == 0) {
      *normalizing_factor = 1.f / Z;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      const float val =
          sycl::exp((static_cast<float>(input[idx]) - (*float_max))) *
          (*normalizing_factor);
      output[idx] = val;
    }
  }

 private:
  sycl::local_accessor<float, 1> slm;
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_cols;
};

template <int TPB, typename IndType>
class moeTopK {
 public:
  moeTopK(
      const float* inputs_after_softmax,
      const bool* finished,
      float* output,
      IndType* indices,
      int* source_rows,
      const int num_experts,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize)
      : inputs_after_softmax(inputs_after_softmax),
        finished(finished),
        output(output),
        indices(indices),
        source_rows(source_rows),
        num_experts(num_experts),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    int kIdx;
    float kVal;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int num_rows = item.get_group_range(0);
    const int block_row = group_id_x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = group_id_x * num_experts;
    float sum_val = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      kIdx = 0;
      kVal = -1.f; // This is OK because inputs are probabilities

      int inpIdx;
      float inpVal;
      for (int expert = local_id_x; expert < num_experts; expert += TPB) {
        const int idx = thread_read_offset + expert;
        inpIdx = expert;
        inpVal = inputs_after_softmax[idx];

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[k * block_row + prior_k];

          if (prior_winning_expert == expert) {
            inpIdx = kIdx;
            inpVal = kVal;
          }
        }

        if (inpVal > kVal) {
          kIdx = inpIdx;
          kVal = inpVal;
        }
      }

      const float resultVal =
          sycl::reduce_over_group(group, kVal, sycl::maximum<float>());
      const int resultIdx = sycl::reduce_over_group(
          group, resultVal == kVal ? kIdx : 0x7FFFFFFF, sycl::minimum<int>());
      sum_val += resultVal;

      if (local_id_x == 0) {
        // Ignore experts the node isn't responsible for with expert parallelism
        const int expert = resultIdx;
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        const int idx = k * block_row + k_idx;
        output[idx] = resultVal;
        indices[idx] =
            should_process_row ? (expert - start_expert) : num_experts;
        assert(indices[idx] >= 0);
        source_rows[idx] = k_idx * num_rows + block_row;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (renormalize) {
      auto local_range_x = item.get_local_range(0);
      for (int k_idx = local_id_x; k_idx < k; k_idx += local_range_x) {
        const int idx = k * block_row + k_idx;
        output[idx] /= sum_val;
      }
    }
  }

 private:
  const float* inputs_after_softmax;
  const bool* finished;
  float* output;
  IndType* indices;
  int* source_rows;
  const int num_experts;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
};

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small
  power of 2. Additionally it also supports when number of experts is multiple
  of 64 which is still faster than the computing softmax and topK separately. 2)
  This implementation assumes k is small, but will work for any k.
*/

template <
    int VPT,
    int NUM_EXPERTS,
    int WARPS_PER_CTA,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType,
    typename IndType>
class topkGatingSoftmax {
 public:
  topkGatingSoftmax(
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_rows,
      IndType* indices,
      int* source_rows,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize)
      : input(input),
        finished(finished),
        output(output),
        num_rows(num_rows),
        indices(indices),
        source_rows(source_rows),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<2> item) const {
    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);
    // We begin by enforcing compile time assertions and setting up compile time
    // constants.
    static_assert(
        BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
        "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(
        VPT % ELTS_PER_LDG == 0,
        "The elements per thread must be a multiple of the elements per ldg");
    static_assert(
        WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
        "The threads per row must cleanly divide the threads per warp");
    static_assert(
        THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
        "THREADS_PER_ROW must be power of 2");
    static_assert(
        THREADS_PER_ROW <= WARP_SIZE_PARAM,
        "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(
        ELTS_PER_WARP % ELTS_PER_ROW == 0,
        "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing
    // run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and
    // a block contains WARPS_PER_CTA warps. This, each block processes a chunk
    // of rows. We start by computing the start row for each block.
    const int cta_base_row = group_id_x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per
    // warp.
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = local_id_x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows) {
      return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First,
    // each thread jumps to the start of the row it will read.
    const InputdType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the
    // first column to start loads.
    const int thread_group_idx = local_id_x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputdType* thread_read_ptr =
        thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    InputdType row_chunk_load[VPT];
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
#pragma unroll
      for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
        row_chunk_load[ii * ELTS_PER_LDG + jj] =
            thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + jj];
      }
    }

    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = static_cast<float>(row_chunk_load[ii]);
    }

    // First, we perform a max reduce within the thread. We can do the max in
    // fp16 safely (I think) and just convert to float afterwards for the exp +
    // sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = MAX(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      thread_max =
          MAX(thread_max, sycl::permute_group_by_xor(sg, thread_max, mask));
    }

    // From this point, thread max in all the threads have the max within the
    // row. Now, we subtract the max from each element in the thread and take
    // the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = sycl::exp(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
    }

    // From this point, all threads have the max and the sum for their rows in
    // the thread_max and thread_sum variables respectively. Finally, we can
    // scale the rows for the softmax. Technically, for top-k gating we don't
    // need to compute the entire softmax row. We can likely look at the maxes
    // and only compute for the top-k values in the row. However, this kernel
    // will likely not be a bottle neck and it seems better to closer match
    // torch and find the argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to
    // find the topk elements in each row, along with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float sum_val = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // First, each thread does the local argmax
      float max_val = row_chunk[0];
      int expert = start_col;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];

          // No check on the experts here since columns with the smallest index
          // are processed first and only updated if > (not >=)
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = sycl::permute_group_by_xor(sg, max_val, mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, mask);

        // We want lower indices to "win" in every thread so we break ties this
        // way
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      sum_val += max_val;

      // Write the max for this k iteration to global memory.
      if (thread_group_idx == 0) {
        // Add a guard to ignore experts not included by this node
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        // The lead thread from each sub-group will write out the final results
        // to global memory. (This will be a single) thread per row of the
        // input/output matrices.
        const int idx = k * thread_row + k_idx;
        output[idx] = max_val;
        indices[idx] =
            should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        source_rows[idx] = k_idx * num_rows + thread_row;
      }

      // Finally, we clear the value in the thread with the current max if there
      // is another iteration to run.
      if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group =
            (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

        // Only the thread in the group which produced the max will reset the
        // "winning" value to -inf.
        if (thread_group_idx == thread_to_clear_in_group) {
          const int offset_for_expert = expert % ELTS_PER_LDG;
          // Safe to set to any negative value since row_chunk values must be
          // between 0 and 1.
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
              -10000.f;
        }
      }
    }

    if (renormalize) {
      for (int k_idx = thread_group_idx; k_idx < k; k_idx += THREADS_PER_ROW) {
        const int idx = k * thread_row + k_idx;
        output[idx] /= sum_val;
      }
    }
  }

 private:
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_rows;
  IndType* indices;
  int* source_rows;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
};

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <
    int EXPERTS,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
  static_assert(
      EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
          EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
      "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
} // namespace detail

template <
    int EXPERTS,
    int WARPS_PER_TB,
    int WARP_SIZE_PARAM,
    int MAX_BYTES_PER_LDG,
    typename InputdType,
    typename IndType>
void topkGatingSoftmaxLauncherHelper(
    const InputdType* input,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_row,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    bool renormalize,
    sycl::queue& queue) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputdType) * EXPERTS);
  using Constants = detail::
      TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputdType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<2> grid(1, num_blocks);
  sycl::range<2> block(WARPS_PER_TB, WARP_SIZE_PARAM);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(grid * block, block),
        topkGatingSoftmax<
            VPT,
            EXPERTS,
            WARPS_PER_TB,
            BYTES_PER_LDG,
            WARP_SIZE_PARAM,
            InputdType,
            IndType>(
            input,
            finished,
            output,
            num_rows,
            indices,
            source_row,
            k,
            start_expert,
            end_expert,
            renormalize));
  });
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                   \
  static_assert(                                                               \
      WARP_SIZE == 32, "Unsupported warp size. Only 32 is supported for XPU"); \
  topkGatingSoftmaxLauncherHelper<                                             \
      NUM_EXPERTS,                                                             \
      WARPS_PER_TB,                                                            \
      WARP_SIZE,                                                               \
      MAX_BYTES>(                                                              \
      gating_output,                                                           \
      nullptr,                                                                 \
      topk_weights,                                                            \
      topk_indices,                                                            \
      token_expert_indices,                                                    \
      num_tokens,                                                              \
      topk,                                                                    \
      0,                                                                       \
      num_experts,                                                             \
      renormalize,                                                             \
      queue);

template <typename InputdType, typename IndType>
void topkGatingSoftmaxKernelLauncher(
    const InputdType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    sycl::queue& queue) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 = 2 * sizeof(InputdType);

  switch (num_experts) {
    case 1:
      LAUNCH_SOFTMAX(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 2:
      LAUNCH_SOFTMAX(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 4:
      LAUNCH_SOFTMAX(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 8:
      LAUNCH_SOFTMAX(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 16:
      LAUNCH_SOFTMAX(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 32:
      LAUNCH_SOFTMAX(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 64:
      LAUNCH_SOFTMAX(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 128:
      LAUNCH_SOFTMAX(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 256:
      LAUNCH_SOFTMAX(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 512:
      LAUNCH_SOFTMAX(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 192:
      LAUNCH_SOFTMAX(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 320:
      LAUNCH_SOFTMAX(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 384:
      LAUNCH_SOFTMAX(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 448:
      LAUNCH_SOFTMAX(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 576:
      LAUNCH_SOFTMAX(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    default: {
      TORCH_CHECK(
          softmax_workspace != nullptr,
          "softmax_workspace must be provided for num_experts that are "
          "not a power of 2 or multiple of 64.");
      static constexpr int TPB = 256;
      sycl::range<1> grid1(num_tokens);
      sycl::range<1> block1(TPB);
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(grid1 * block1, block1),
            moeSoftmax<TPB, InputdType>(
                slm, gating_output, nullptr, softmax_workspace, num_experts));
      });

      sycl::range<1> grid2(num_tokens);
      sycl::range<1> block2(TPB);
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(grid2 * block2, block2),
            moeTopK<TPB, IndType>(
                softmax_workspace,
                nullptr,
                topk_weights,
                topk_indices,
                token_expert_indices,
                num_experts,
                topk,
                0,
                num_experts,
                renormalize));
      });
    }
  }
}
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
        local_elems[e] = Sigmoid(s);
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
          continue;
        }
        topk_ids[offset + i] = topk_ids_local[i];
      }
    }
  }
  float* topk_weights;
  int* topk_ids;
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

template <int TopK, typename T>
struct MoEScatter {
  static constexpr int GroupWorkItem = 256;
  static constexpr int SgSize = 16;
  static constexpr int NumSg = GroupWorkItem / SgSize;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  static constexpr int32_t EXCLUSIVE_SIZE = 1024;

  MoEScatter(
      sycl::local_accessor<int32_t, 1>& slm,
      const T* activations,
      const int* rows_for_experts,
      const int* topk_indices,
      const int* offsets,
      T* reordered_activation,
      int* mapped_slot,
      const int n_tokens,
      const int experts_offset,
      const int n_experts_local,
      const int n_channels)
      : slm(slm),
        activations(activations),
        rows_for_experts(rows_for_experts),
        topk_indices(topk_indices),
        offsets(offsets),
        reordered_activation(reordered_activation),
        mapped_slot(mapped_slot),
        n_tokens(n_tokens),
        experts_offset(experts_offset),
        n_experts_local(n_experts_local),
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
    auto local_range = item.get_local_range(0);

    int32_t* expert_cumsum_ptr = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    for (int i = local_id; i < n_experts_local; i += local_range) {
      expert_cumsum_ptr[i] = rows_for_experts[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + EXCLUSIVE_SIZE,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    int indices[TopK], expert_local_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      indices[i] = topk_indices[token_id * TopK + i];
      expert_local_offset[i] = offsets[token_id * TopK + i];
    }

    int expert_row_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      if (expert_local_offset[i] == -1) {
        expert_row_offset[i] = -1;
        continue;
      }
      int expert_id = indices[i] - experts_offset;
      int start_offset = expert_id == 0 ? 0 : expert_cumsum_ptr[expert_id - 1];
      expert_row_offset[i] = start_offset + expert_local_offset[i];
    }

    const T* activation_base =
        activations + token_id * n_channels + local_id * ElemsPerItem;
    T* reordered_activation_bases[TopK];
    for (int i = 0; i < TopK; ++i) {
      if (expert_local_offset[i] == -1) {
        reordered_activation_bases[i] = nullptr;
      } else {
        reordered_activation_bases[i] = reordered_activation +
            expert_row_offset[i] * n_channels + local_id * ElemsPerItem;
      }
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
          if (expert_local_offset[i] == -1)
            continue;
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
  sycl::local_accessor<int32_t, 1> slm;
  const T* activations; // [n_tokens, n_channels]
  const int* rows_for_experts; // [n_experts]
  const int* topk_indices; // [n_tokens, num_top_k]
  const int* offsets; // [n_tokens, num_top_k]
  T* reordered_activation;
  int* mapped_slot; // [n_tokens, num_top_k]
  const int n_tokens;
  const int experts_offset;
  const int n_experts_local;
  const int n_channels;
};

template <int TopK>
struct MoEScatter<TopK, sycl::ext::oneapi::bfloat16> {
  using T = sycl::ext::oneapi::bfloat16;
  static constexpr int GroupWorkItem = 256;
  static constexpr int SgSize = 16;
  static constexpr int NumSg = GroupWorkItem / SgSize;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  static constexpr int32_t EXCLUSIVE_SIZE = 1024;

  MoEScatter(
      sycl::local_accessor<int32_t, 1>& slm,
      const T* activations,
      const int* rows_for_experts,
      const int* topk_indices,
      const int* offsets,
      T* reordered_activation,
      int* mapped_slot,
      const int n_tokens,
      const int experts_offset,
      const int n_experts_local,
      const int n_channels)
      : slm(slm),
        activations(activations),
        rows_for_experts(rows_for_experts),
        topk_indices(topk_indices),
        offsets(offsets),
        reordered_activation(reordered_activation),
        mapped_slot(mapped_slot),
        n_tokens(n_tokens),
        experts_offset(experts_offset),
        n_experts_local(n_experts_local),
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
    auto local_range = item.get_local_range(0);

    int32_t* expert_cumsum_ptr = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    for (int i = local_id; i < n_experts_local; i += local_range) {
      expert_cumsum_ptr[i] = rows_for_experts[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + EXCLUSIVE_SIZE,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    int indices[TopK], expert_local_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      indices[i] = topk_indices[token_id * TopK + i];
      expert_local_offset[i] = offsets[token_id * TopK + i];
    }

    int expert_row_offset[TopK];
    for (int i = 0; i < TopK; ++i) {
      if (expert_local_offset[i] == -1) {
        expert_row_offset[i] = -1;
        continue;
      }
      int expert_id = indices[i] - experts_offset;
      int start_offset = expert_id == 0 ? 0 : expert_cumsum_ptr[expert_id - 1];
      expert_row_offset[i] = start_offset + expert_local_offset[i];
    }

    const T* activation_base =
        activations + token_id * n_channels + local_id * ElemsPerItem;
    T* reordered_activation_bases[TopK];
    for (int i = 0; i < TopK; ++i) {
      if (expert_local_offset[i] == -1) {
        reordered_activation_bases[i] = nullptr;
      } else {
        reordered_activation_bases[i] = reordered_activation +
            expert_row_offset[i] * n_channels + local_id * ElemsPerItem;
      }
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
          if (expert_local_offset[i] == -1)
            continue;
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
  sycl::local_accessor<int32_t, 1> slm;
  const T* activations; // [n_tokens, n_channels]
  const int* rows_for_experts; // [n_experts]
  const int* topk_indices; // [n_tokens, num_top_k]
  const int* offsets; // [n_tokens, num_top_k]
  T* reordered_activation;
  int* mapped_slot; // [n_tokens, num_top_k]
  const int n_tokens;
  const int experts_offset;
  const int n_experts_local;
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
    const int experts_offset,
    const int n_experts_local,
    const int n_channels) {
  using Kernel = MoEScatter<TopK, T>;
  // TODO: maybe add template for GroupWorkItem in the future
  TORCH_CHECK(
      Kernel::EXCLUSIVE_SIZE >= n_experts_local,
      "MoEScatter::EXCLUSIVE_SIZE is expected to be larger than num_expert");
  TORCH_CHECK(
      n_channels % Kernel::ElemsPerItem == 0,
      "n_channels is expected to be aligned to Kernel::ElemsPerItem");

  auto range = Kernel::get_nd_range(n_tokens);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(Kernel::EXCLUSIVE_SIZE), cgh);
    Kernel task(
        slm,
        activations,
        rows_for_experts,
        topk_indices,
        offsets,
        reordered_activation,
        mapped_slot,
        n_tokens,
        experts_offset,
        n_experts_local,
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
    const int experts_offset,
    const int n_experts_local,
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
        experts_offset,       \
        n_experts_local,      \
        n_channels);          \
    break;
  switch (n_topk) {
    CASE_TOPK(1)
    CASE_TOPK(2)
    CASE_TOPK(4)
    CASE_TOPK(6)
    CASE_TOPK(8)
    CASE_TOPK(10)
    default:
      TORCH_CHECK(false, "error: not support topk=%d,\n", topk);
  }
#undef CASE_TOPK
};
}; // namespace MoEScatterImpl

namespace MoEGatherImpl {
// Re-gather the outputs of MoE and scale them by the gating score.
template <int TopK, typename T>
struct MoEGather {
  static constexpr int GroupWorkItem = 256;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);
  static constexpr int Stride = ElemsPerItem * GroupWorkItem;
  MoEGather(
      T* layer_output,
      const T* moe_output,
      const float* scores,
      const int32_t* mapped_slots,
      const int32_t n_channels,
      const int32_t n_experts,
      const int32_t n_tokens,
      const bool normalize_scales)
      : layer_output(layer_output),
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
      if (token_mapped_slots[i] == -1)
        continue;
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
          if (token_mapped_slots[j] == -1)
            continue;
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
      const T* moe_output,
      const float* scores,
      const int32_t* mapped_slots,
      const int32_t n_channels,
      const int32_t n_experts,
      const int32_t n_tokens,
      const bool normalize_scales)
      : layer_output(layer_output),
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
      if (token_mapped_slots[i] == -1)
        continue;
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
          if (token_mapped_slots[j] == -1)
            continue;
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
    const int32_t n_channels,
    const int32_t n_experts,
    const int32_t n_tokens,
    const bool normalize_scales) {
  using Kernel = MoEGather<TopK, T>;
  TORCH_CHECK(
      n_channels % Kernel::ElemsPerItem == 0,
      "n_channels is expected to be aligned to Kernel::ElemsPerItem");
  auto range = Kernel::get_nd_range(n_tokens);
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        layer_output,
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
    CASE_TOPK(10)
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

void topk_softmax(
    at::Tensor& topk_weights, // [num_tokens, topk]
    at::Tensor& topk_indices, // [num_tokens, topk]
    at::Tensor& token_expert_indices, // [num_tokens, topk]
    at::Tensor& gating_output, // [num_tokens, num_experts]
    const bool renormalize) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  auto& queue = dpcppGetCurrentQueue();
  at::Tensor softmax_workspace =
      at::empty({workspace_size}, gating_output.options().dtype(at::kFloat));

#define LAUNCH_TOPK_SOFTMAX(INPUTDTYPE, INDTYPE)                       \
  TopKSoftmaxImpl::topkGatingSoftmaxKernelLauncher(                    \
      reinterpret_cast<INPUTDTYPE*>(gating_output.mutable_data_ptr()), \
      topk_weights.data_ptr<float>(),                                  \
      topk_indices.data_ptr<INDTYPE>(),                                \
      token_expert_indices.data_ptr<int>(),                            \
      softmax_workspace.data_ptr<float>(),                             \
      num_tokens,                                                      \
      num_experts,                                                     \
      topk,                                                            \
      renormalize,                                                     \
      queue);

  if (topk_indices.scalar_type() == at::ScalarType::Int) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, int)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, int)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, int)
  } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, uint32_t)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, uint32_t)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, uint32_t)
  } else {
    TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, int64_t)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, int64_t)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, int64_t)
  }

#undef LAUNCH_TOPK_SOFTMAX
}

/**
 * @brief Perform grouped topk after sigmoid/addbias on gating_output.
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param n_topk The number of top experts to select.
 * @param n_topk_group The number of top experts to select in the group.
 * @return A tuple of tensors (topk_weights, topk_indices, rows_for_experts,
 * offsets).
 */
static std::tuple<at::Tensor, at::Tensor> grouped_topk(
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

  if (gating_output.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
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
    const int64_t experts_offset,
    const int64_t n_experts_local,
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
        experts_offset,
        n_experts_local,
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
        experts_offset,
        n_experts_local,
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
              experts_offset,
              n_experts_local,
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
  IPEX_OP_REGISTER("moe_sum.moe", at::AtenIpexTypeXPU::moe_sum);
}
} // namespace
