#include <ATen/native/CPUBlas.h>
#include <aten/utils/common.h>
#include <aten/utils/vec.h>
#include <aten/utils/amx.h>
#include <aten/utils/woq.h>
#include <aten/utils/gemm.h>
#include <aten/DSMoE.h>
#include <cassert>
#include "vec/vec.h"
namespace torch_ipex {
namespace cpu {

namespace {
at::Tensor call_AllReduce(const at::Tensor& self) {
  static auto op_allreduce =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("deepspeed_comm::all_reduce", "")
          .typed<at::Tensor(const at::Tensor& self)>();
  auto ret = op_allreduce.call(self);
  return ret;
}

// [NOTE]: Fused MoE kernel with AMX
//
//   This file contains implementations for
//     * `moe_align_block_size`
//     * `fused_moe`
//
//   The functionality is identical to triton kernel, excepts:
//     * fuse silu_and_mul with gemm1, therefore this kernel
//       allocates 2 intermediate_caches instead of 3
//     * add `offsets` in `moe_align_block_size` which keeps track
//       of starting offset for each M block. this is for keeping
//       output of silu_and_mul in sorted order, thus load_A for
//       the 2nd gemm would be contiguous, therefore we can directly
//       load A from intermediate_cache1.
//
//  TODO:
//     1. tune BLOCK_M and BLOCK_N (BLOCK_N * K fit L2)
//     2. add prefetch for load A which is indexed access
//     3. abstract at::native::cpublas::brgemm with WoQ gemm (M = 1 & M != 1)
//

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, scalar_t val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec data_vec(val);
  at::vec::map<scalar_t>(
      [data_vec](Vec out) { return out = data_vec; }, out, out, size);
}
template <typename scalar_t>
inline void copy_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
// no remainder
#pragma GCC unroll 4
  for (int d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <typename scalar_t>
inline void copy_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}
template <typename scalar_t>
inline void copy_mul_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    float weight,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]*weight);
  }
}
template <typename scalar_t>
inline void copy_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float weight,
    int size) {
  float input_[size];
  cvt_bf16_to_fp32(input_, input, size);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input_ + d) * weight_vec;
    fVec data1 = fVec::loadu(input_ + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input_[d]*weight);
  }
}
// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int topk,
    int K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts
  // accumulate count of expert ids locally
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);
    for (int i = begin; i < end; ++i) {
      local_cnts[topk_ids[i]]++;
    }
  });
  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; },
        T_INDEX(t + 1),
        T_INDEX(t + 1),
        T_INDEX(t),
        num_experts);
  }
  // the last row holds sums of each experts
  int32_t* total_cnts_t_1 = T_INDEX(num_threads);
  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    // accumulate `num_tokens_post_pad`, also as the expert offset
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    // thread tid offsets in `total_cnts`
    int32_t* __restrict__ offsets = T_INDEX(tid);
    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      offsets[expert_id]++;
    }
  });
  // debug: the offset for thread t_1 should be identical to t_2
  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e) {
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);
  }
  // padding value for sorted_ids: numel
  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };
  // offsets holds starting offset for each valida M blocks
  //   shape : [num_token_blocks + 1]
  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(
      0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
        for (int mb = begin; mb < end; ++mb) {
          offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
        }
      });
  // TODO: do we need to vecterize this ?
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }
  // debug: the last value of offsets should be `numel`
  TORCH_CHECK(offsets[num_token_blocks] == numel);
  return num_tokens_post_pad;
}

template <int BLOCK_M>
int moe_align_block_size_topk1(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts
  // accumulate count of expert ids locally
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);
    for (int i = begin; i < end; ++i) {
      local_cnts[0]++;
    }
  });
  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; },
        T_INDEX(t + 1),
        T_INDEX(t + 1),
        T_INDEX(t),
        num_experts);
  }
  // the last row holds sums of each experts
  int32_t* total_cnts_t_1 = T_INDEX(num_threads);
  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    // accumulate `num_tokens_post_pad`, also as the expert offset
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    // thread tid offsets in `total_cnts`
    int32_t* __restrict__ offsets = T_INDEX(tid);
    for (int i = begin; i < end; ++i) {
      int32_t expert_id = 0;
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      offsets[expert_id]++;
    }
  });
  // debug: the offset for thread t_1 should be identical to t_2
  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e) {
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);
  }
  // padding value for sorted_ids: numel
  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };
  // offsets holds starting offset for each valida M blocks
  //   shape : [num_token_blocks + 1]
  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(
      0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
        for (int mb = begin; mb < end; ++mb) {
          offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
        }
      });
  // TODO: do we need to vecterize this ?
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }
  // debug: the last value of offsets should be `numel`
  TORCH_CHECK(offsets[num_token_blocks] == numel);
  return num_tokens_post_pad;
}

//   silu :    shape          leading dimension
//  input0  [m_size, BLOCK_N]    BLOCK_N
//  input1  [m_size, BLOCK_N]    BLOCK_N
//  output  [M * topk, N]          N
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ input0, // x: x0, x1
    const float* __restrict__ input1, // y: y0, y1
    int m_size,
    int N) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  // no remainder
  for (int m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ x = input0 + m * BLOCK_N;
    const float* __restrict__ y = input1 + m * BLOCK_N;
    for (int d = 0; d < BLOCK_N; d += bVec::size()) {
      fVec x0 = fVec::loadu(x + d);
      fVec x1 = fVec::loadu(x + d + fVec::size());
      fVec y0 = fVec::loadu(y + d);
      fVec y1 = fVec::loadu(y + d + fVec::size());
      // silu
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
      // mul
      x0 = x0 * y0;
      x1 = x1 * y1;
      // convert
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(out + d);
    }
  }
}
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input0, // x: x0, x1
    const scalar_t* __restrict__ input1, // y: y0, y1
    int m_size,
    int N) {
  float input0_[m_size*N];
  cvt_bf16_to_fp32(input0_, input0, m_size*N);
  float input1_[m_size*N];
  cvt_bf16_to_fp32(input1_, input1, m_size*N);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  // no remainder
  for (int m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ x = input0_ + m * BLOCK_N;
    const float* __restrict__ y = input0_ + m * BLOCK_N;
    for (int d = 0; d < BLOCK_N; d += bVec::size()) {
      fVec x0 = fVec::loadu(x + d);
      fVec x1 = fVec::loadu(x + d + fVec::size());
      fVec y0 = fVec::loadu(y + d);
      fVec y1 = fVec::loadu(y + d + fVec::size());
      // silu
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
      // mul
      x0 = x0 * y0;
      x1 = x1 * y1;
      // convert
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(out + d);
    }
  }
}

 void Dequantize_and_compute(
  uint8_t* qB,
  long M,
  long K,
  long N,
  at::BFloat16* scales,
  at::BFloat16* zps,
  const at::BFloat16* __restrict__ act,
  float* __restrict__ out,
  long ldb,
  long N_GROUP_SIZE,
  bool is_woq_sym) {
    // std::cout<<"----0----:"<<out[0]<<std::endl;
  #if defined(CPU_CAPABILITY_AVX512_BF16)
  using T = at::BFloat16;
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  // lookup table converting uint8 to float, 15.0f - 0.0f
  // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
  // table to honor the lower 4 bits regardless of the the highest bit, thus
  // saving an "and" op
  VT lut;
  lut = V::set_0_to_15();
  dequant_n_grouped_and_compute(qB, M, K, N, scales, zps, act, out, ldb, N_GROUP_SIZE, is_woq_sym);

  #endif
  }
template <typename scalar_t>
void fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ C_tmp,
    float* __restrict__ C_tmp_f,
    const scalar_t* __restrict__ input,
    at::Tensor& packed_w1_tensor,
    at::Tensor& packed_w2_tensor,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int M,
    int N,
    int K,
    int E,
    int topk,
    int num_tokens_post_pad,
    bool is_woq,
    const bool is_woq_sym,
    scalar_t* w1_scale,
    scalar_t* w1_zp,
    scalar_t* w2_scale,
    scalar_t* w2_zp) {
  // handle 2 tiles per block
  uint8_t* packed_qw1 = nullptr;
  uint8_t* packed_qw2 = nullptr;
  scalar_t* packed_w1 = nullptr;
  scalar_t* packed_w2 = nullptr;
  if (!is_woq) {
    packed_w1 = (scalar_t*)packed_w1_tensor.data_ptr<scalar_t>();
    packed_w2 = (scalar_t*)packed_w2_tensor.data_ptr<scalar_t>();
  } else {
    packed_qw1 = (uint8_t*)packed_w1_tensor.data_ptr<int8_t>();
    packed_qw2 = (uint8_t*)packed_w2_tensor.data_ptr<int8_t>();
  }

  constexpr int BLOCK_M = block_size_m();
  constexpr int T_BLOCK_N = block_size_n(); // Tuned block_n for AMX
  constexpr int Q_BLOCK_N = WOQ_N_BLOCK_SIZE; // Tuned block_n for WOQ
  int BLOCK_N = is_woq ? Q_BLOCK_N : T_BLOCK_N;
  int Q_BLOCK_K = is_woq ? packed_w2_tensor.size(3) : -1;
  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  // strides for w1: [E, 2N, K]
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);
  TORCH_CHECK(
      K % Q_BLOCK_K == 0, "Fixme when K is not multiples of ", Q_BLOCK_K);
  const int stride_e = 2 * N * K;
  const int stride_n = K;
  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
    // get local pointers
    int tid = at::get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float*  C0_f = C_tmp_f + tid * 2 * BLOCK_M * BLOCK_N;
    float*  C1_f = C0_f + BLOCK_M * BLOCK_N;
    for (int i = begin; i < end; ++i) {
      int mb = i / NB;
      int nb = i % NB;
      // nb0 from top half and nb1 from bottom half
      int nb0 = nb, nb1 = nb + NB;
      int n_size = std::min(N - nb0 * BLOCK_N, BLOCK_N);

      int32_t expert_id = expert_ids[mb];

      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int m_size = offsets[mb + 1] - offsets[mb];
      bool use_brgemm = m_size > 1;
      for (int m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      if(use_brgemm){
        torch::Tensor dequant_packed_w1_0 = torch::empty(
          {K / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        torch::Tensor dequant_packed_w1_1 = torch::empty(
          {K / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        if (is_woq) { // Dequant loop
          uint8_t* qB0 =
              packed_qw1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          scalar_t* w1_scale_0 = w1_scale + expert_id * 2* N + nb0 * BLOCK_N;
          scalar_t* w1_zp_0 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb0 * BLOCK_N;
          uint8_t* qB1 =
              packed_qw1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          scalar_t* w1_scale_1 = w1_scale + expert_id * 2* N + nb1 * BLOCK_N;
          scalar_t* w1_zp_1 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb1 * BLOCK_N;

          for (int k_i = 0; k_i < K; k_i = k_i + Q_BLOCK_K) {
            if(is_woq_sym){
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  true, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB0 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_0,
                      w1_zp_0,
                      dequant_packed_w1_0.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }else{
              Dequantize<
              scalar_t, // target : bf16  gjn here dequant
              Q_BLOCK_N,
              get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
              WOQ_DTYPE_INT8, // qw_type_
              false, // sym_quant_w
              false>:: // use_g_idx
              call(
                  qB0 + k_i * BLOCK_N,
                  Q_BLOCK_K,
                  n_size,
                  w1_scale_0,
                  w1_zp_0,
                  dequant_packed_w1_0.data_ptr<scalar_t>() + k_i * BLOCK_N,
                  0,
                  nullptr); // g_idx_ptr
            }
          }
          for (int k_i = 0; k_i < K; k_i = k_i + Q_BLOCK_K) {
            if(is_woq_sym){
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  true, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB1 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_1,
                      w1_zp_1,
                      dequant_packed_w1_1.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }else{
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  false, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB1 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_1,
                      w1_zp_1,
                      dequant_packed_w1_1.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }
          }
        }
        const scalar_t* __restrict__ B0 = is_woq
        ? dequant_packed_w1_0.data_ptr<scalar_t>()
        : packed_w1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
        const scalar_t* __restrict__ B1 = is_woq
        ? dequant_packed_w1_1.data_ptr<scalar_t>()
        : packed_w1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B0,
            /* C     */ C0_f);
        // 1.c gemm: C1 = A @ B1
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B1,
            /* C     */ C1_f);
      }else{
        if (is_woq) { // Dequant loop
          uint8_t* qB0 =
              packed_qw1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          scalar_t* w1_scale_0 = w1_scale + expert_id * 2* N + nb0 * BLOCK_N;
          scalar_t* w1_zp_0 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb0 * BLOCK_N;
          // 2.a gemm: C = A @ B
            Dequantize_and_compute(
                    qB0 ,
                    m_size,
                    K,
                    n_size,
                    w1_scale_0,
                    w1_zp_0,
                    A,
                    C0_f,
                    Q_BLOCK_N,
                    get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                    is_woq_sym);
          uint8_t* qB1 =
              packed_qw1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          scalar_t* w1_scale_1 = w1_scale + expert_id * 2* N + nb1 * BLOCK_N;
          scalar_t* w1_zp_1 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb1 * BLOCK_N;
            Dequantize_and_compute(
                    qB1 ,
                    m_size,
                    K,
                    n_size,
                    w1_scale_1,
                    w1_zp_1,
                    A,
                    C1_f,
                    Q_BLOCK_N,
                    get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                    is_woq_sym);

        }else{
          const scalar_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          torch_ipex::cpu::tinygemm_kernel<scalar_t, scalar_t>(
            /*   A */ A,
            /*   B */ B0 /* nb * BLOCK_N * K */,
            /*   C */ C0_f,
            /*scale*/ 0.f,
            /*   M */ m_size,
            /*   N */ n_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ n_size,
            /* ldc */ BLOCK_N);
          const scalar_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          torch_ipex::cpu::tinygemm_kernel<scalar_t, scalar_t>(
            /*   A */ A,
            /*   B */ B1 /* nb * BLOCK_N * K */,
            /*   C */ C1_f,
            /*scale*/ 0.f,
            /*   M */ m_size,
            /*   N */ n_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ n_size,
            /* ldc */ BLOCK_N);
        }
      }
 
      const int offset = offsets[mb];
      if (is_woq) {
        silu_and_mul<scalar_t, Q_BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N, C0_f, C1_f, m_size, N);
      } else {
        silu_and_mul<scalar_t, T_BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N, C0_f, C1_f, m_size, N);
      }
      if(use_brgemm){
        at::native::cpublas::brgemm_release();
      }
    }

  });
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int OC = K; // rename K as OC
  const int IC = N; // rename N as IC
  const int MB2 = MB;
  const int NB2 = div_up(OC, BLOCK_N);
  const int stride_e2 = OC * IC;
  const int stride_oc = IC;
  TORCH_CHECK(
      IC % Q_BLOCK_K == 0, "Fixme when K is not multiples of ", Q_BLOCK_K);
  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int begin, int end) {
    // get local pointers
    int tid = at::get_thread_num();
    // we won't be using C1 for gemm2
    float*  C_f = C_tmp_f + tid * 2 * BLOCK_M * BLOCK_N;
    for (int i = begin; i < end; ++i) {
      int mb = i / NB2;
      int nb = i % NB2;
      int m_size = offsets[mb + 1] - offsets[mb];
      bool use_brgemm = m_size > 1;
      int n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);
      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t*  A = ic1 + offsets[mb] * N; // + nb * BLOCK_N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      if(use_brgemm){
        torch::Tensor dequant_packed_w2 = torch::empty(
          {IC / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        if (is_woq) { // Dequant loop
          uint8_t* qB = packed_qw2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          scalar_t* w2_zp_ = is_woq_sym? nullptr : w2_zp + expert_id * OC + nb * BLOCK_N;
          scalar_t* w2_scale_ = w2_scale + expert_id * OC + nb * BLOCK_N;
          for (int k_i = 0; k_i < IC; k_i = k_i + Q_BLOCK_K) {
            if(is_woq_sym){
              Dequantize<
              scalar_t, // dequant dtype
              Q_BLOCK_N,
              get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
              WOQ_DTYPE_INT8, // qw_type_
              true, // sym_quant_w
              false>:: // use_g_idx
              call(
                qB + k_i * BLOCK_N,
                  Q_BLOCK_K,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  dequant_packed_w2.data_ptr<scalar_t>() + k_i * BLOCK_N,
                  0,
                  nullptr); // g_idx_ptr
            }else{
              Dequantize<
              scalar_t, // dequant dtype
              Q_BLOCK_N,
              get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
              WOQ_DTYPE_INT8, // qw_type_
              false, // sym_quant_w
              false>:: // use_g_idx
              call(
                qB + k_i * BLOCK_N,
                  Q_BLOCK_K,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  dequant_packed_w2.data_ptr<scalar_t>() + k_i * BLOCK_N,
                  0,
                  nullptr); // g_idx_ptr
            }
          }
        }
        const scalar_t* __restrict__ B = is_woq
        ? dequant_packed_w2.data_ptr<scalar_t>()
        : packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B,
            /* C     */ C_f);
      }else{
        if (is_woq) { // Dequant loop
          uint8_t* qB =
              packed_qw2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          scalar_t* w2_zp_ = is_woq_sym? nullptr : w2_zp + expert_id * OC + nb * BLOCK_N;
          scalar_t* w2_scale_ = w2_scale + expert_id * OC + nb * BLOCK_N;
          // 2.a gemm: C = A @ B
          Dequantize_and_compute(
                  qB ,
                  m_size,
                  IC,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  A,
                  C_f,
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  is_woq_sym);
        }else{
          const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          torch_ipex::cpu::tinygemm_kernel<at::BFloat16, at::BFloat16>(
              /*   A */ A,
              /*   B */ B /* nb * BLOCK_N * K */,
              /*   C */ C_f,
              /*scale*/ 0.f,
              /*   M */ m_size,
              /*   N */ n_size,
              /*   K */ IC,
              /* lda */ IC,
              /* ldb */ n_size,
              /* ldc */ BLOCK_N);
        }
      }

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(
            ic2 + index * K + nb * BLOCK_N, C_f + m * BLOCK_N, weight, n_size);
      }
      if(use_brgemm){
        at::native::cpublas::brgemm_release();
      }
    }

  });
  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int begin, int end) {
    for (int m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

// hidden_states: [M, K]
// w1: [E, 2N, K]
// w2: [E, K, N]
// topk_weights: [M, topk]
// topk_ids: [M, topk] (int32_t)
//
at::Tensor fused_experts_impl(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    bool is_woq_sym,
    at::Tensor w1_scale,
    at::Tensor w1_zp,
    at::Tensor w2_scale,
    at::Tensor w2_zp) {
  assert(is_vnni == true);
  auto packed_w1 = w1;
  auto packed_w2 = w2;
  constexpr int BLOCK_M = block_size_m();
  int BLOCK_N = is_woq ? WOQ_N_BLOCK_SIZE : block_size_n();
  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());
  CHECK_DIM(2, hidden_states);
  if (!is_woq) {
    CHECK_DIM(3, w1);
    CHECK_DIM(3, w2);
  } else {
    // {E, N/block_n, K/block_k, block_k, block_n}
    CHECK_DIM(5, w1);
    CHECK_DIM(5, w2);
  }
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_EQ(topk_ids.scalar_type(), at::kInt);
  CHECK_EQ(topk_weights.scalar_type(), at::kFloat);
  int M = hidden_states.size(0);
  int K = hidden_states.size(1);
  int N = is_woq ? w2.size(2) * w2.size(3) : w2.size(2);
  int E = w1.size(0);
  int topk = topk_weights.size(1);
  // check weight shapes
  if (!is_woq) {
    CHECK_EQ(w1.size(1), 2 * N);
    CHECK_EQ(w1.size(2), K);
    CHECK_EQ(w2.size(0), E);
    CHECK_EQ(w2.size(1), K);
  }
  at::Tensor out_hidden_states =
      inplace ? hidden_states : at::empty_like(hidden_states);
  // NB: worst case is each expert holds a block with remainder of 1
  //   1. sorted_ids : [M * topk + E * (BLOCK_M - 1)]
  //   2. expert_ids : [max_num_blocks]
  //   3. total_cnts : [T + 1, E]
  //   4. cumsums    : [E + 1]
  //   5. offsets    : [max_num_blocks + 1]
  //
  int num_threads = at::get_num_threads();
  int max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  auto buffer = at::empty(
      {max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E +
       (E + 1) + (max_num_blocks + 1)},
      topk_ids.options());
  int32_t* __restrict__ sorted_ids = buffer.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);
  // init sorted_ids with `numel` as the padding number
  // init expert_ids with `num_experts`
  int numel = M * topk;
  at::parallel_for(
      0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
        int m_start = begin * BLOCK_M;
        int m_size =
            std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
        fill_stub(sorted_ids + m_start, numel, m_size);
        fill_stub(expert_ids + begin, E, end - begin);
      });
  // zero total_cnts and cumsums
  at::parallel_for(
      0, (num_threads + 1) * E + (E + 1), GRAIN_SIZE, [&](int begin, int end) {
        fill_stub(total_cnts + begin, 0, end - begin);
      });
  // align experts index
  int num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
      sorted_ids,
      expert_ids,
      topk_ids.data_ptr<int32_t>(),
      total_cnts,
      cumsums,
      offsets,
      E,
      numel,
      num_threads);
  // unlike triton kernel, we fuse silu with gemm1 so only need 2
  // intermediate_caches:
  //   1. intermediate_cache1 : [M * topk, N]
  //   2. intermediate_cache2 : [M * topk, K]
  //   3. A_tmp : [T, BLOCK_M * K]
  //   4. C_tmp : [T, 2 * BLOCK_M * BLOCK_N] x 2
  //
  auto buffer2 = at::empty(
      {M * topk * N + M * topk * K + num_threads * BLOCK_M * K +
       num_threads * 2 * BLOCK_M * BLOCK_N *
           /* sizeof(float) / sizeof(scalar_t) */ 2},
      hidden_states.options());
  using scalar_t = c10::BFloat16;
  // AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_experts_kernel_impl", [&] {
  scalar_t* __restrict__ intermediate_cache1 = buffer2.data_ptr<scalar_t>();
  scalar_t* __restrict__ intermediate_cache2 =
      intermediate_cache1 + M * topk * N;
  scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
  scalar_t* __restrict__ C_tmp =
      (scalar_t*)((void*)(A_tmp + num_threads * BLOCK_M * K));
  float* __restrict__ C_tmp_f =
      (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
  fused_experts_kernel_impl<scalar_t>(
      out_hidden_states.data_ptr<scalar_t>(),
      intermediate_cache1,
      intermediate_cache2,
      A_tmp,
      C_tmp,
      C_tmp_f,
      hidden_states.data_ptr<scalar_t>(),
      packed_w1,
      packed_w2,
      topk_weights.data_ptr<float>(),
      sorted_ids,
      expert_ids,
      offsets,
      M,
      N,
      K,
      E,
      topk,
      num_tokens_post_pad,
      is_woq,
      is_woq_sym,
      w1_scale.data_ptr<scalar_t>(),
      w1_zp.data_ptr<scalar_t>(),
      w2_scale.data_ptr<scalar_t>(),
      w2_zp.data_ptr<scalar_t>());
  // });
  if (is_distributed) {
    call_AllReduce(out_hidden_states);
  }
  return out_hidden_states;
}


template <typename scalar_t>
void fused_mlp_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ C_tmp,
    float* __restrict__ C_tmp_f,
    const scalar_t* __restrict__ input,
    at::Tensor& packed_w1_tensor,
    at::Tensor& packed_w2_tensor,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int M,
    int N,
    int K,
    int E,
    int topk,
    int num_tokens_post_pad,
    bool is_woq,
    const bool is_woq_sym,
    scalar_t* w1_scale,
    scalar_t* w1_zp,
    scalar_t* w2_scale,
    scalar_t* w2_zp) {
  // handle 2 tiles per block
  uint8_t* packed_qw1 = nullptr;
  uint8_t* packed_qw2 = nullptr;
  scalar_t* packed_w1 = nullptr;
  scalar_t* packed_w2 = nullptr;
  if (!is_woq) {
    packed_w1 = (scalar_t*)packed_w1_tensor.data_ptr<scalar_t>();
    packed_w2 = (scalar_t*)packed_w2_tensor.data_ptr<scalar_t>();
  } else {
    packed_qw1 = (uint8_t*)packed_w1_tensor.data_ptr<int8_t>();
    packed_qw2 = (uint8_t*)packed_w2_tensor.data_ptr<int8_t>();
  }

  constexpr int BLOCK_M = block_size_m();
  constexpr int T_BLOCK_N = block_size_n(); // Tuned block_n for AMX
  constexpr int Q_BLOCK_N = WOQ_N_BLOCK_SIZE; // Tuned block_n for WOQ
  int BLOCK_N = is_woq ? Q_BLOCK_N : T_BLOCK_N;
  int Q_BLOCK_K = is_woq ? packed_w2_tensor.size(3) : -1;
  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  // strides for w1: [E, 2N, K]
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);
  TORCH_CHECK(
      K % Q_BLOCK_K == 0, "Fixme when K is not multiples of ", Q_BLOCK_K);
  const int stride_e = 2 * N * K;
  const int stride_n = K;
  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
    // get local pointers
    int tid = at::get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float*  C0_f = C_tmp_f + tid * 2 * BLOCK_M * BLOCK_N;
    float*  C1_f = C0_f + BLOCK_M * BLOCK_N;
    for (int i = begin; i < end; ++i) {
      int mb = i / NB;
      int nb = i % NB;
      // nb0 from top half and nb1 from bottom half
      int nb0 = nb, nb1 = nb + NB;
      int n_size = std::min(N - nb0 * BLOCK_N, BLOCK_N);

      int32_t expert_id = expert_ids[mb];

      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int m_size = offsets[mb + 1] - offsets[mb];
      bool use_brgemm = m_size > 1;
      for (int m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      if(use_brgemm){
        torch::Tensor dequant_packed_w1_0 = torch::empty(
          {K / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        torch::Tensor dequant_packed_w1_1 = torch::empty(
          {K / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        if (is_woq) { // Dequant loop
          uint8_t* qB0 =
              packed_qw1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          scalar_t* w1_scale_0 = w1_scale + expert_id * 2* N + nb0 * BLOCK_N;
          scalar_t* w1_zp_0 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb0 * BLOCK_N;
          uint8_t* qB1 =
              packed_qw1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          scalar_t* w1_scale_1 = w1_scale + expert_id * 2* N + nb1 * BLOCK_N;
          scalar_t* w1_zp_1 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb1 * BLOCK_N;

          for (int k_i = 0; k_i < K; k_i = k_i + Q_BLOCK_K) {
            if(is_woq_sym){
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  true, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB0 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_0,
                      w1_zp_0,
                      dequant_packed_w1_0.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }else{
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  false, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB0 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_0,
                      w1_zp_0,
                      dequant_packed_w1_0.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }
          }
          for (int k_i = 0; k_i < K; k_i = k_i + Q_BLOCK_K) {
            if(is_woq_sym){
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  true, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB1 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_1,
                      w1_zp_1,
                      dequant_packed_w1_1.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }else{
              Dequantize<
                  scalar_t, // target : bf16  gjn here dequant
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
                  WOQ_DTYPE_INT8, // qw_type_
                  false, // sym_quant_w
                  false>:: // use_g_idx
                  call(
                      qB1 + k_i * BLOCK_N,
                      Q_BLOCK_K,
                      n_size,
                      w1_scale_1,
                      w1_zp_1,
                      dequant_packed_w1_1.data_ptr<scalar_t>() + k_i * BLOCK_N,
                      0,
                      nullptr); // g_idx_ptr
            }
          }
        }
        const scalar_t* __restrict__ B0 = is_woq
        ? dequant_packed_w1_0.data_ptr<scalar_t>()
        : packed_w1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
        const scalar_t* __restrict__ B1 = is_woq
        ? dequant_packed_w1_1.data_ptr<scalar_t>()
        : packed_w1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B0,
            /* C     */ C0_f);
        // 1.c gemm: C1 = A @ B1
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B1,
            /* C     */ C1_f);
      }else{
        if (is_woq) { // Dequant loop
          uint8_t* qB0 =
              packed_qw1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          scalar_t* w1_scale_0 = w1_scale + expert_id * 2* N + nb0 * BLOCK_N;
          scalar_t* w1_zp_0 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb0 * BLOCK_N;
          // 2.a gemm: C = A @ B
            Dequantize_and_compute(
                    qB0 ,
                    m_size,
                    K,
                    n_size,
                    w1_scale_0,
                    w1_zp_0,
                    A,
                    C0_f,
                    Q_BLOCK_N,
                    get_n_group_size(Q_BLOCK_N),
                    is_woq_sym); // N_GROUP_SIZE
          uint8_t* qB1 =
              packed_qw1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          scalar_t* w1_scale_1 = w1_scale + expert_id * 2* N + nb1 * BLOCK_N;
          scalar_t* w1_zp_1 = is_woq_sym? nullptr : w1_zp + expert_id * 2* N + nb1 * BLOCK_N;
            Dequantize_and_compute(
                    qB1 ,
                    m_size,
                    K,
                    n_size,
                    w1_scale_1,
                    w1_zp_1,
                    A,
                    C1_f,
                    Q_BLOCK_N,
                    get_n_group_size(Q_BLOCK_N),
                    is_woq_sym); // N_GROUP_SIZE

        }else{
          const scalar_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb0 * BLOCK_N * stride_n;
          torch_ipex::cpu::tinygemm_kernel<scalar_t, scalar_t>(
            /*   A */ A,
            /*   B */ B0 /* nb * BLOCK_N * K */,
            /*   C */ C0_f,
            /*scale*/ 0.f,
            /*   M */ m_size,
            /*   N */ n_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ n_size,
            /* ldc */ BLOCK_N);
          const scalar_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb1 * BLOCK_N * stride_n;
          torch_ipex::cpu::tinygemm_kernel<scalar_t, scalar_t>(
            /*   A */ A,
            /*   B */ B1 /* nb * BLOCK_N * K */,
            /*   C */ C1_f,
            /*scale*/ 0.f,
            /*   M */ m_size,
            /*   N */ n_size,
            /*   K */ K,
            /* lda */ K,
            /* ldb */ n_size,
            /* ldc */ BLOCK_N);
        }
      }
 
      const int offset = offsets[mb];
      if (is_woq) {
        silu_and_mul<scalar_t, Q_BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N, C0_f, C1_f, m_size, N);
      } else {
        silu_and_mul<scalar_t, T_BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N, C0_f, C1_f, m_size, N);
      }
      if(use_brgemm){
        at::native::cpublas::brgemm_release();
      }
    }

  });
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int OC = K; // rename K as OC
  const int IC = N; // rename N as IC
  const int MB2 = MB;
  const int NB2 = div_up(OC, BLOCK_N);
  const int stride_e2 = OC * IC;
  const int stride_oc = IC;
  TORCH_CHECK(
      IC % Q_BLOCK_K == 0, "Fixme when K is not multiples of ", Q_BLOCK_K);
  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int begin, int end) {
    // get local pointers
    int tid = at::get_thread_num();
    // we won't be using C1 for gemm2
    float*  C_f = C_tmp_f + tid * 2 * BLOCK_M * BLOCK_N;
    for (int i = begin; i < end; ++i) {
      int mb = i / NB2;
      int nb = i % NB2;
      int m_size = offsets[mb + 1] - offsets[mb];
      bool use_brgemm = m_size > 1;
      int n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);
      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t*  A = ic1 + offsets[mb] * N; // + nb * BLOCK_N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      if(use_brgemm){
        torch::Tensor dequant_packed_w2 = torch::empty(
          {IC / 2, BLOCK_N, 2}, c10::CppTypeToScalarType<scalar_t>::value);
        if (is_woq) { // Dequant loop
          uint8_t* qB = packed_qw2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          scalar_t* w2_zp_ = is_woq_sym? nullptr : w2_zp + expert_id * OC + nb * BLOCK_N;
          scalar_t* w2_scale_ = w2_scale + expert_id * OC + nb * BLOCK_N;
          for (int k_i = 0; k_i < IC; k_i = k_i + Q_BLOCK_K) {
            if (is_woq_sym){
              Dequantize<
              scalar_t, // dequant dtype
              Q_BLOCK_N,
              get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
              WOQ_DTYPE_INT8, // qw_type_
              true, // sym_quant_w
              false>:: // use_g_idx
              call(
                qB + k_i * BLOCK_N,
                  Q_BLOCK_K,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  dequant_packed_w2.data_ptr<scalar_t>() + k_i * BLOCK_N,
                  0,
                  nullptr); // g_idx_ptr
            }else{
              Dequantize<
              scalar_t, // dequant dtype
              Q_BLOCK_N,
              get_n_group_size(Q_BLOCK_N), // N_GROUP_SIZE
              WOQ_DTYPE_INT8, // qw_type_
              false, // sym_quant_w
              false>:: // use_g_idx
              call(
                qB + k_i * BLOCK_N,
                  Q_BLOCK_K,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  dequant_packed_w2.data_ptr<scalar_t>() + k_i * BLOCK_N,
                  0,
                  nullptr); // g_idx_ptr
            }
          }
        }
        const scalar_t* __restrict__ B = is_woq
        ? dequant_packed_w2.data_ptr<scalar_t>()
        : packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B,
            /* C     */ C_f);
      }else{
        if (is_woq) { // Dequant loop
          uint8_t* qB =
              packed_qw2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          scalar_t* w2_zp_ = is_woq_sym? nullptr : w2_zp + expert_id * OC + nb * BLOCK_N;
          scalar_t* w2_scale_ = w2_scale + expert_id * OC + nb * BLOCK_N;
          // 2.a gemm: C = A @ B
          Dequantize_and_compute(
                  qB ,
                  m_size,
                  IC,
                  n_size,
                  w2_scale_,
                  w2_zp_,
                  A,
                  C_f,
                  Q_BLOCK_N,
                  get_n_group_size(Q_BLOCK_N),
                  is_woq_sym); // N_GROUP_SIZE
        }else{
          const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
          torch_ipex::cpu::tinygemm_kernel<at::BFloat16, at::BFloat16>(
              /*   A */ A,
              /*   B */ B /* nb * BLOCK_N * K */,
              /*   C */ C_f,
              /*scale*/ 0.f,
              /*   M */ m_size,
              /*   N */ n_size,
              /*   K */ IC,
              /* lda */ IC,
              /* ldb */ n_size,
              /* ldc */ BLOCK_N);
        }
      }

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        copy_stub(
          output + index * K + nb * BLOCK_N, C_f + m * BLOCK_N, n_size);
      }
      if(use_brgemm){
        at::native::cpublas::brgemm_release();
      }
    }

  });
  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  // at::parallel_for(0, M, 0, [&](int begin, int end) {
  //   for (int m = begin; m < end; ++m) {
  //     sum_stub(output + m * K, ic2 + m  * K, 1, K);
  //   }
  // });
}

// hidden_states: [M, K]
// w1: [E, 2N, K]
// w2: [E, K, N]
// topk_weights: [M, topk]
// topk_ids: [M, topk] (int32_t)
//
at::Tensor fused_mlp_impl(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    bool is_woq_sym,
    at::Tensor w1_scale,
    at::Tensor w1_zp,
    at::Tensor w2_scale,
    at::Tensor w2_zp) {
  assert(is_vnni == true);
  auto packed_w1 = w1;
  auto packed_w2 = w2;
  constexpr int BLOCK_M = block_size_m();
  int BLOCK_N = is_woq ? WOQ_N_BLOCK_SIZE : block_size_n();
  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  if (!is_woq) {
    CHECK_DIM(3, w1);
    CHECK_DIM(3, w2);
  } else {
    // {E, N/block_n, K/block_k, block_k, block_n}
    CHECK_DIM(5, w1);
    CHECK_DIM(5, w2);
  }
  int M = hidden_states.size(0);
  int K = hidden_states.size(1);
  int N = is_woq ? w2.size(2) * w2.size(3) : w2.size(2);
  int E = w1.size(0);
  int topk = 1;
  // check weight shapes
  if (!is_woq) {
    CHECK_EQ(w1.size(1), 2 * N);
    CHECK_EQ(w1.size(2), K);
    CHECK_EQ(w2.size(0), E);
    CHECK_EQ(w2.size(1), K);
  }
  at::Tensor out_hidden_states =
      inplace ? hidden_states : at::empty_like(hidden_states);
  // NB: worst case is each expert holds a block with remainder of 1
  //   1. sorted_ids : [M * topk + E * (BLOCK_M - 1)]
  //   2. expert_ids : [max_num_blocks]
  //   3. total_cnts : [T + 1, E]
  //   4. cumsums    : [E + 1]
  //   5. offsets    : [max_num_blocks + 1]
  //
  int num_threads = at::get_num_threads();
  int max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  auto buffer = at::empty(
      {max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E +
       (E + 1) + (max_num_blocks + 1)},
       at::kInt);
  int32_t* __restrict__ sorted_ids = buffer.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);
  // init sorted_ids with `numel` as the padding number
  // init expert_ids with `num_experts`
  int numel = M * topk;
  at::parallel_for(
      0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
        int m_start = begin * BLOCK_M;
        int m_size =
            std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
        fill_stub(sorted_ids + m_start, numel, m_size);
        fill_stub(expert_ids + begin, E, end - begin);
      });
  // zero total_cnts and cumsums
  at::parallel_for(
      0, (num_threads + 1) * E + (E + 1), GRAIN_SIZE, [&](int begin, int end) {
        fill_stub(total_cnts + begin, 0, end - begin);
      });
  // align experts index
  int num_tokens_post_pad = moe_align_block_size_topk1<BLOCK_M>(
      sorted_ids,
      expert_ids,
      total_cnts,
      cumsums,
      offsets,
      E,
      numel,
      num_threads);
  // unlike triton kernel, we fuse silu with gemm1 so only need 2
  // intermediate_caches:
  //   1. intermediate_cache1 : [M * topk, N]
  //   2. intermediate_cache2 : [M * topk, K]
  //   3. A_tmp : [T, BLOCK_M * K]
  //   4. C_tmp : [T, 2 * BLOCK_M * BLOCK_N] x 2

  auto buffer2 = at::empty(
      {M * topk * N + M * topk * K + num_threads * BLOCK_M * K +
       num_threads * 2 * BLOCK_M * BLOCK_N *
           /* sizeof(float) / sizeof(scalar_t) */ 2},
      hidden_states.options());
  using scalar_t = c10::BFloat16;
  // AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_experts_kernel_impl", [&] {
  scalar_t* __restrict__ intermediate_cache1 = buffer2.data_ptr<scalar_t>();
  scalar_t* __restrict__ intermediate_cache2 =
      intermediate_cache1 + M * topk * N;
  scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
  scalar_t* __restrict__ C_tmp =
      (scalar_t*)((void*)(A_tmp + num_threads * BLOCK_M * K));
  float* __restrict__ C_tmp_f =
      (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
  fused_mlp_kernel_impl<scalar_t>(
      out_hidden_states.data_ptr<scalar_t>(),
      intermediate_cache1,
      intermediate_cache2,
      A_tmp,
      C_tmp,
      C_tmp_f,
      hidden_states.data_ptr<scalar_t>(),
      packed_w1,
      packed_w2,
      sorted_ids,
      expert_ids,
      offsets,
      M,
      N,
      K,
      E,
      topk,
      num_tokens_post_pad,
      is_woq,
      is_woq_sym,
      w1_scale.data_ptr<scalar_t>(),
      w1_zp.data_ptr<scalar_t>(),
      w2_scale.data_ptr<scalar_t>(),
      w2_zp.data_ptr<scalar_t>());

  if (is_distributed) {
    call_AllReduce(out_hidden_states);
  }
  return out_hidden_states;
}



} // anonymous namespace

IPEX_REGISTER_DISPATCH(fused_experts_impl_stub, &fused_experts_impl);
IPEX_REGISTER_DISPATCH(fused_mlp_impl_stub, &fused_mlp_impl);
} // namespace cpu
} // namespace torch_ipex
