#include "DSMoE.h"
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <aten/utils/amx.h>
#include <aten/utils/common.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(fused_experts_impl_stub);
template <typename T>
inline void copy_and_fill(
    T* __restrict__ out,
    const T* __restrict__ input,
    int size,
    int pad_size,
    T fill_value) {
  using Vec = at::vec::Vectorized<T>;
  int d = 0;
  if (size >= Vec::size()) {
#pragma GCC unroll 4
    for (; d < size; d += Vec::size()) {
      Vec data = Vec::loadu(input + d);
      data.store(out + d);
    }
  }
  for (; d < size; ++d) {
    out[d] = input[d];
  }
  // using scalar padding as pad_size is less than vec size here
  for (; d < pad_size; ++d) {
    out[d] = fill_value;
  }
}

at::Tensor fused_experts_with_shared(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation) {
  RECORD_FUNCTION(
      "ipex::fused_experts_with_shared", c10::ArrayRef<c10::IValue>({}));
  int32_t num_tokens = topk_weights.size(0);
  int32_t num_topk_experts = topk_weights.size(1);
  int32_t num_topk_experts_pad = num_topk_experts + 1;
  int32_t num_experts = w1.size(0);
  auto pad_weight =
      at::empty({num_tokens, num_topk_experts_pad}, topk_weights.options());
  auto pad_ids =
      at::empty({num_tokens, num_topk_experts_pad}, topk_ids.options());
  // padding 1 shared expert to routed expert
  // topk_id is num_experts - 1, and topk weights is 1.0
  for (int id = 0; id < num_tokens; id++) {
    copy_and_fill<int32_t>(
        pad_ids.data_ptr<int32_t>() + id * num_topk_experts_pad,
        topk_ids.data_ptr<int32_t>() + id * num_topk_experts,
        num_topk_experts,
        num_topk_experts_pad,
        num_experts - 1);
    copy_and_fill<float>(
        pad_weight.data_ptr<float>() + id * num_topk_experts_pad,
        topk_weights.data_ptr<float>() + id * num_topk_experts,
        num_topk_experts,
        num_topk_experts_pad,
        1.0);
  }
  return fused_experts_impl_stub(
      kCPU,
      hidden_states,
      w1,
      w2,
      pad_weight,
      pad_ids,
      inplace,
      is_vnni,
      is_distributed,
      is_woq,
      woq_weight_dtype,
      woq_group_size,
      woq_lowp_mode,
      w1_scale,
      w1_zp,
      w1_compensation,
      w2_scale,
      w2_zp,
      w2_compensation);
}

at::Tensor fused_experts(
    const at::Tensor& hidden_states,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    int64_t woq_weight_dtype,
    int64_t woq_group_size,
    int64_t woq_lowp_mode,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w1_zp,
    const std::optional<at::Tensor>& w1_compensation,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w2_zp,
    const std::optional<at::Tensor>& w2_compensation) {
  RECORD_FUNCTION("ipex::fused_experts", c10::ArrayRef<c10::IValue>({}));

  return fused_experts_impl_stub(
      kCPU,
      hidden_states,
      w1,
      w2,
      topk_weights,
      topk_ids,
      inplace,
      is_vnni,
      is_distributed,
      is_woq,
      woq_weight_dtype,
      woq_group_size,
      woq_lowp_mode,
      w1_scale,
      w1_zp,
      w1_compensation,
      w2_scale,
      w2_zp,
      w2_compensation);
}

constexpr int block_size_m() {
  return 1 * TILE_M;
}
constexpr int block_size_n() {
  return 8 * TILE_N;
}
constexpr int block_size_n2() {
  return 2 * TILE_N;
}
// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
//
// [N, K/2, 2] to [K/2, N, 2]
template <typename scalar_t>
inline void pack_vnni(
    scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ weight,
    int N,
    int K) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] =
            weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> convert_weight_packed_moe_bf16(
    at::Tensor& w1,
    at::Tensor& w2) {
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_DIM(3, w1);
  CHECK_DIM(3, w2);
  const auto st1 = w1.scalar_type();
  const auto st2 = w2.scalar_type();
  TORCH_CHECK(st1 == st2, "weight type mismatch");
  const int E1 = w1.size(0);
  const int E2 = w2.size(0);
  const int OC1 = w1.size(1);
  const int IC1 = w1.size(2);
  const int OC2 = w2.size(1);
  const int IC2 = w2.size(2);
  TORCH_CHECK(E1 == E2 && IC2 * 2 == OC1 && IC1 == OC2, "weight size mismatch");
  const int N = IC2;
  const int K = IC1;
  // we handle 2 TILE_N at a time.
  TORCH_CHECK(N % TILE_N == 0, "invalid weight out features ", N);
  TORCH_CHECK(K % TILE_K == 0, "invalid weight input features ", K);
  int BLOCK_N = block_size_n();
  if (N < 2 * BLOCK_N) {
    BLOCK_N = block_size_n2();
  }

  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight1 = at::empty({E1, OC1, IC1}, w1.options());
  auto packed_weight2 = at::empty({E2, OC2, IC2}, w2.options());
  const int stride1 = OC1 * IC1;
  const int stride2 = OC2 * IC2;
  // TODO: add float8 support
  TORCH_CHECK(
      st1 == at::kBFloat16 || st1 == at::kHalf,
      "expect weight to be bfloat16 or float16.");
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st1, "conver_weight_packed_impl", [&] {
    const scalar_t* w_data1 = w1.data_ptr<scalar_t>();
    const scalar_t* w_data2 = w2.data_ptr<scalar_t>();
    scalar_t* packed_data1 = packed_weight1.data_ptr<scalar_t>();
    scalar_t* packed_data2 = packed_weight2.data_ptr<scalar_t>();
    // parallel on {E}
    at::parallel_for(0, E1, 0, [&](int begin, int end) {
      for (int e = begin; e < end; ++e) {
        for (int n = 0; n < OC1; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, OC1 - n);
          pack_vnni<scalar_t>(
              packed_data1 + e * stride1 + n * IC1,
              w_data1 + e * stride1 + n * IC1,
              n_size,
              IC1);
        }
      }
    });
    at::parallel_for(0, E2, 0, [&](int begin, int end) {
      for (int e = begin; e < end; ++e) {
        for (int n = 0; n < OC2; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, OC2 - n);
          pack_vnni<scalar_t>(
              packed_data2 + e * stride2 + n * IC2,
              w_data2 + e * stride2 + n * IC2,
              n_size,
              IC2);
        }
      }
    });
  });

  return std::make_tuple(packed_weight1, packed_weight2);
}

template <typename scalar_t, int SIZE>
inline void sigmoid(
    float* __restrict__ out,
    const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  // step 0: convert input
  fVec one_fvec = fVec(1.0);
  if constexpr (SIZE < kVecSize) {
    // SIZE = 1, 2, 4, 8, 16; only the top half is used
    bVec x_bvec = bVec::loadu(input, SIZE);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
    x_fvec0.store(out, SIZE);
  } else {
    for (int d = 0; d < SIZE; d += kVecSize) {
      bVec x_bvec = bVec::loadu(input + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
      x_fvec0.store(out + d);
      x_fvec1.store(out + d + fVec::size());
    }
  }

  fVec zero_fvec = fVec(0.0);
  // div_out = (1 + (-x).exp())
  // out = 1/ div_out
  if constexpr (SIZE < fVec::size()) {
    // SIZE = 1, 2, 4, 8
    fVec x_fvec =
        one_fvec / (one_fvec + (zero_fvec - fVec::loadu(out, SIZE)).exp_u20());
    x_fvec.store(out, SIZE);
  } else {
    for (int d = 0; d < SIZE; d += fVec::size()) {
      fVec x_fvec =
          one_fvec / (one_fvec + (zero_fvec - fVec::loadu(out + d)).exp_u20());
      x_fvec.store(out + d);
    }
  }
}
template <typename scalar_t, int NUM_EXPERTS>
void grouped_topk_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    int num_tokens,
    int topk,
    int num_groups,
    int topk_group,
    bool renormalize,
    float* __restrict__ e_score_correction_bias,
    float* routed_scaling_factor) {
  const int num_experts_per_group = NUM_EXPERTS / num_groups;
  parallel_for(num_tokens, [&](int begin, int end) {
    static thread_local float scores[NUM_EXPERTS];
    static thread_local float ori_scores[NUM_EXPERTS];
    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue_temp(num_groups);
    std::vector<elem_t> queue(num_groups);
    std::vector<elem_t> queue2(topk_group * num_experts_per_group);

    for (int i = begin; i < end; ++i) {
      // do softmax to get scores
      sigmoid<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      for (int g = 0; g < NUM_EXPERTS; ++g) {
        ori_scores[g] = scores[g];
        scores[g] = scores[g] + e_score_correction_bias[g];
      }
      // find max score per group
      for (int g = 0; g < num_groups; ++g) {
        float gmax = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < num_experts_per_group; ++e) {
          gmax = std::max(gmax, scores[g * num_experts_per_group + e]);
        }
        queue_temp[g] = {gmax, g};
      }
      for (int g = 0; g < num_groups; ++g) {
        float pervious_max = queue_temp[g].first;
        int count_pervious_max = 1;
        float gmax = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < num_experts_per_group; ++e) {
          if (count_pervious_max == 1 &&
              scores[g * num_experts_per_group + e] == pervious_max) {
            count_pervious_max--;
          } else {
            gmax = std::max(gmax, scores[g * num_experts_per_group + e]);
          }
        }
        queue[g] = {gmax + pervious_max, g};
      }
      // find group topk
      std::partial_sort(
          queue.begin(),
          queue.begin() + topk_group,
          queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int g = 0; g < topk_group; ++g) {
        int32_t group_idx = queue[g].second;
        for (int e = 0; e < num_experts_per_group; ++e) {
          int32_t expert_idx = group_idx * num_experts_per_group + e;
          queue2[g * num_experts_per_group + e] = {
              scores[expert_idx], expert_idx};
        }
      }
      // find global topk
      std::partial_sort(
          queue2.begin(),
          queue2.begin() + topk,
          queue2.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });
      for (int j = 0; j < topk; ++j) {
        topk_weights[i * topk + j] = ori_scores[queue2[j].second];
        topk_ids[i * topk + j] = queue2[j].second;
      }
      if (renormalize) {
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        for (int j = 0; j < topk; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
      }
      for (int j = 0; j < topk; ++j) {
        topk_weights[i * topk + j] =
            topk_weights[i * topk + j] * routed_scaling_factor[0];
      }
    }
  });
}

#define LAUNCH_GROUPED_TOPK_KERNEL(NE)           \
  grouped_topk_kernel_impl<at::BFloat16, NE>(    \
      topk_weights.data_ptr<float>(),            \
      topk_ids.data_ptr<int32_t>(),              \
      gating_output.data_ptr<at::BFloat16>(),    \
      num_tokens,                                \
      topk,                                      \
      num_expert_group,                          \
      topk_group,                                \
      renormalize,                               \
      e_score_correction_bias.data_ptr<float>(), \
      routed_scaling_factor.data_ptr<float>());

#define LAUNCH_GROUPED_TOPK_KERNEL_FP16(NE)      \
  grouped_topk_kernel_impl<at::Half, NE>(        \
      topk_weights.data_ptr<float>(),            \
      topk_ids.data_ptr<int32_t>(),              \
      gating_output.data_ptr<at::Half>(),        \
      num_tokens,                                \
      topk,                                      \
      num_expert_group,                          \
      topk_group,                                \
      renormalize,                               \
      e_score_correction_bias.data_ptr<float>(), \
      routed_scaling_factor.data_ptr<float>());
//
std::tuple<at::Tensor, at::Tensor> grouped_topk(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    at::Tensor& e_score_correction_bias,
    at::Tensor& routed_scaling_factor) {
  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  auto topk_weights = at::empty({num_tokens, topk}, at::kFloat);
  auto topk_ids = at::empty_like(topk_weights, at::kInt);
  if (st == at::kBFloat16) {
    switch (num_experts) {
      case 1:
        LAUNCH_GROUPED_TOPK_KERNEL(1);
        break;
      case 2:
        LAUNCH_GROUPED_TOPK_KERNEL(2);
        break;
      case 4:
        LAUNCH_GROUPED_TOPK_KERNEL(4);
        break;
      case 8:
        LAUNCH_GROUPED_TOPK_KERNEL(8);
        break;
      case 16:
        LAUNCH_GROUPED_TOPK_KERNEL(16);
        break;
      case 32:
        LAUNCH_GROUPED_TOPK_KERNEL(32);
        break;
      case 64:
        LAUNCH_GROUPED_TOPK_KERNEL(64);
        break;
      case 128:
        LAUNCH_GROUPED_TOPK_KERNEL(128);
        break;
      case 256:
        LAUNCH_GROUPED_TOPK_KERNEL(256);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  } else if (st == at::kHalf) {
    switch (num_experts) {
      case 1:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(1);
        break;
      case 2:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(2);
        break;
      case 4:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(4);
        break;
      case 8:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(8);
        break;
      case 16:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(16);
        break;
      case 32:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(32);
        break;
      case 64:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(64);
        break;
      case 128:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(128);
        break;
      case 256:
        LAUNCH_GROUPED_TOPK_KERNEL_FP16(256);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  }
  return std::make_tuple(topk_ids, topk_weights);
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fused_experts(Tensor hidden_states, Tensor w1, Tensor w2, Tensor topk_weights, \
       Tensor topk_ids, bool inplace, bool is_vnni, \
       bool is_distributed, bool is_woq, int woq_weight_dtype, int woq_group_size, int woq_lowp_mode, \
       Tensor? w1_scale, Tensor? w1_zp, Tensor? w1_compensation, Tensor? w2_scale, Tensor? w2_zp, Tensor? w2_compensation) -> Tensor");
  m.impl(
      "fused_experts", c10::DispatchKey::CPU, torch_ipex::cpu::fused_experts);
  m.def(
      "fused_experts_with_shared(Tensor hidden_states, Tensor w1, Tensor w2, Tensor topk_weights, \
         Tensor topk_ids, bool inplace, bool is_vnni, \
         bool is_distributed, bool is_woq, int woq_weight_dtype, int woq_group_size, int woq_lowp_mode, \
         Tensor? w1_scale, Tensor? w1_zp, Tensor? w1_compensation, Tensor? w2_scale, Tensor? w2_zp, Tensor? w2_compensation) -> Tensor");
  m.impl(
      "fused_experts_with_shared",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::fused_experts_with_shared);
  m.def(
      "grouped_topk(Tensor hidden_states, Tensor gating_output, \
        int topk, bool renormalize, int num_expert_group, int topk_group, Tensor e_score_correction_bias, Tensor routed_scaling_factor)  -> (Tensor, Tensor)");
  m.impl("grouped_topk", c10::DispatchKey::CPU, torch_ipex::cpu::grouped_topk);
  m.def(
      "convert_weight_packed_moe_bf16(Tensor weight1, Tensor weight2) -> (Tensor, Tensor)");
  m.impl(
      "convert_weight_packed_moe_bf16",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convert_weight_packed_moe_bf16);
}
} // namespace