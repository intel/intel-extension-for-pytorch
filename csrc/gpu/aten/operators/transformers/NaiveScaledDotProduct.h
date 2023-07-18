#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <torch/library.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "../RandomEngine.h"
#include "../comm/ATDispatch.h"
#include "../comm/AccumulateType.h"
#include "../comm/ApplyUtils.h"
#include "../comm/Numerics.h"
#include "../comm/TensorOptions.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

namespace memory {

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
  T val[vec_size];
};

template <typename scalar_t, int BLOCK_Q, int BLOCK_KV, int HEAD_SIZE>
struct alignas(32) fmha_shared_memory {
  enum {
    BLOCK_SIZE_Q = BLOCK_Q * HEAD_SIZE,
    BLOCK_SIZE_KV = BLOCK_KV * HEAD_SIZE,
    BLOCK_SIZE_ATTN = BLOCK_Q * BLOCK_KV,
  };
  scalar_t q[BLOCK_SIZE_Q];
  union {
    scalar_t k[BLOCK_SIZE_KV];
    scalar_t v[BLOCK_SIZE_KV];
  };
  union {
    scalar_t qk[BLOCK_SIZE_ATTN];
    scalar_t p[BLOCK_SIZE_ATTN];
  };
  scalar_t m_prev[BLOCK_Q];
  scalar_t l_prev[BLOCK_Q];
  scalar_t m_curr[BLOCK_Q];
  scalar_t l_curr[BLOCK_Q];
  scalar_t l_rcp[BLOCK_Q];
};

} // namespace memory

namespace group {

template <int BLOCK_THREADS>
constexpr int get_threads_per_block_x() {
  if constexpr (BLOCK_THREADS == 1024)
    return 32;
  else if constexpr (BLOCK_THREADS == 256)
    return 16;
  else if constexpr (BLOCK_THREADS == 64)
    return 8;
  else
    return 0;
}

template <
    typename scalar_t,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int BLOCK_THREADS,
    bool A_ROW_MAJOR,
    bool B_ROW_MAJOR,
    bool C_ROW_MAJOR,
    bool ALLOW_ACC_OUT,
    bool FMHA_AUTO_BETA_AND_VEC_WR,
    bool IS_CAUSAL,
    typename item_t,
    typename scalar_tq>
inline void fmha_naive_gemm(
    item_t& item,
    scalar_t* out,
    const scalar_tq* a,
    const scalar_t* b,
    scalar_t alpha,
    scalar_t beta,
    scalar_t* l_rcp = nullptr,
    scalar_t* l_prev = nullptr,
    int row_start = 0,
    int col_start = 0) {
  // out = alpha * (a @ b) + beta * out
  // a: BLOCK_M x BLOCK_K
  // b: BLOCK_K x BLOCK_N
  // out: BLOCK_M x BLOCK_N

  constexpr int THREADS_PER_BLOCK_X = get_threads_per_block_x<BLOCK_THREADS>();
  static_assert(BLOCK_M % THREADS_PER_BLOCK_X == 0);
  static_assert(BLOCK_N % THREADS_PER_BLOCK_X == 0);
  static_assert(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_X == BLOCK_THREADS);
  constexpr int TM = BLOCK_M / THREADS_PER_BLOCK_X;
  constexpr int TN = BLOCK_N / THREADS_PER_BLOCK_X;
  constexpr int TK = 2;
  // constexpr int TM_COUNT = BLOCK_M / TM;
  constexpr int TN_COUNT = BLOCK_N / TN;

  auto lid = item.get_local_id(0);

  auto m_base = lid / TN_COUNT * TM;
  auto n_base = lid % TN_COUNT * TN;

  if constexpr (IS_CAUSAL) {
    static_assert(C_ROW_MAJOR == true);
    row_start += m_base;
    if (row_start < col_start) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          out[(m_base + m) * BLOCK_N + n_base + n] = -1e20;
        }
      }
      return;
    }
    col_start += n_base;
  }

  const scalar_tq* a_base_ptr;
  if constexpr (A_ROW_MAJOR) {
    a_base_ptr = a + m_base * BLOCK_K;
  } else {
    a_base_ptr = a + m_base;
  }

  const scalar_t* b_base_ptr;
  if constexpr (B_ROW_MAJOR) {
    b_base_ptr = b + n_base;
  } else {
    b_base_ptr = b + n_base * BLOCK_K;
  }

  scalar_t a_reg[TM * TK];
  scalar_t b_reg[TK * TN];
  scalar_t acc[TM * TN] = {0.0};

#pragma unroll
  for (int bk = 0; bk < BLOCK_K; bk += TK) {
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int k = 0; k < TK; k++) {
        if constexpr (A_ROW_MAJOR) {
          if constexpr (FMHA_AUTO_BETA_AND_VEC_WR) {
            a_reg[m * TK + k] =
                a_base_ptr[m * BLOCK_K + bk + k] * l_rcp[m_base + m];
          } else {
            a_reg[m * TK + k] = a_base_ptr[m * BLOCK_K + bk + k];
          }
        } else {
          a_reg[m * TK + k] = a_base_ptr[(bk + k) * BLOCK_M + m];
        }
      }
    }
#pragma unroll
    for (int n = 0; n < TN; n++) {
#pragma unroll
      for (int k = 0; k < TK; k++) {
        if constexpr (B_ROW_MAJOR) {
          b_reg[k * TN + n] = b_base_ptr[(bk + k) * BLOCK_N + n];
        } else {
          b_reg[k * TN + n] = b_base_ptr[n * BLOCK_K + bk + k];
        }
      }
    }

#pragma unroll
    for (int k = 0; k < TK; k++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        auto b_ = b_reg[k * TN + n];
#pragma unroll
        for (int m = 0; m < TM; m++) {
          acc[m * TN + n] += a_reg[m * TK + k] * b_;
        }
      }
    }
  }
  if constexpr (!FMHA_AUTO_BETA_AND_VEC_WR) {
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        int idx;
        if constexpr (C_ROW_MAJOR) {
          idx = (m_base + m) * BLOCK_N + n_base + n;
        } else {
          idx = (n_base + n) * BLOCK_M + m_base + m;
        }

        if constexpr (IS_CAUSAL) {
          if (row_start + m < col_start + n) {
            out[idx] = -1e20;
          } else {
            out[idx] = alpha * acc[m * TN + n];
          }
        } else {
          if constexpr (!ALLOW_ACC_OUT) {
            out[idx] = alpha * acc[m * TN + n];
          } else {
            out[idx] = alpha * acc[m * TN + n] + beta * out[idx];
          }
        }
      }
    }
  } else {
    static_assert(C_ROW_MAJOR == true);
    static_assert(ALLOW_ACC_OUT == true);
    using vec_t = memory::aligned_array<scalar_t, TN>;
#pragma unroll
    for (int m = 0; m < TM; m++) {
      vec_t vec;
      int idx = (m_base + m) * BLOCK_N + n_base;
      vec_t out_vec = *reinterpret_cast<vec_t*>(out + idx);
#pragma unroll
      for (int n = 0; n < TN; n++) {
        vec.val[n] = alpha * acc[m * TN + n] +
            out_vec.val[n] * l_rcp[m_base + m] * l_prev[m_base + m];
      }
      *reinterpret_cast<vec_t*>(out + idx) = vec;
    }
  }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N, typename item_t>
inline void fmha_reduce_max_x_sync(
    item_t& item,
    scalar_t* out,
    const scalar_t* block_mat,
    const scalar_t* prev) {
  const scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
  auto m = item.get_local_id(0);
  if (m < BLOCK_M) {
    scalar_t acc = neg_inf;
#pragma unroll
    for (int n = 0; n < BLOCK_N; n++) {
      acc = std::max(acc, block_mat[m * BLOCK_N + n]);
    }
    out[m] = std::max(acc, prev[m]);
  }
  item.barrier(dpcpp_local_fence);
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N, typename item_t>
inline void fmha_reduce_sum_x_sync(
    item_t& item,
    scalar_t* out,
    const scalar_t* block_mat,
    const scalar_t* prev) {
  auto m = item.get_local_id(0);
  if (m < BLOCK_M) {
    scalar_t acc = 0.f;
#pragma unroll
    for (int n = 0; n < BLOCK_N; n++) {
      acc = acc + block_mat[m * BLOCK_N + n];
    }
    out[m] = acc + prev[m];
  }
  item.barrier(dpcpp_local_fence);
}

} // namespace group

template <
    typename scalar_t,
    int BLOCK_Q,
    int BLOCK_KV,
    int HEAD_SIZE,
    int BLOCK_THREADS,
    int VEC_SIZE,
    int WARP_SIZE,
    bool IS_CAUSAL,
    typename item_t>
inline void scaled_dot_product_attention_kernel(
    item_t& item,
    scalar_t* out,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const int q_seq_length,
    const int kv_seq_length,
    scalar_t* out_m,
    scalar_t* out_l,
    char* slm_) {
  auto thread_idx = item.get_local_id(0);
  using vec_t = memory::aligned_array<scalar_t, VEC_SIZE>;
  using slm_t =
      memory::fmha_shared_memory<scalar_t, BLOCK_Q, BLOCK_KV, HEAD_SIZE>;
  auto slm = reinterpret_cast<slm_t*>(slm_);
  auto q_blocks = q_seq_length / BLOCK_Q;
  auto block_idx_0 = item.get_group(0) / q_blocks;
  auto block_idx_1 = item.get_group(0) % q_blocks;

  // calculate ptr base
  auto offset_qo_s =
      (block_idx_0 * q_seq_length + block_idx_1 * BLOCK_Q) * HEAD_SIZE;
  auto o_bs = out + offset_qo_s;
  auto q_bs = const_cast<scalar_t*>(q + offset_qo_s);
  auto offset_kv_b = block_idx_0 * kv_seq_length * HEAD_SIZE;
  auto k_b = const_cast<scalar_t*>(k + offset_kv_b);
  auto v_b = const_cast<scalar_t*>(v + offset_kv_b);
  auto offset_ml_bs = block_idx_0 * q_seq_length + block_idx_1 * BLOCK_Q;
  auto o_ms = out_m + offset_ml_bs;
  auto o_ls = out_l + offset_ml_bs;

  if (thread_idx < BLOCK_Q) {
    slm->m_prev[thread_idx] = -1e20;
    slm->l_prev[thread_idx] = 0;
  }

  constexpr int HEAD_SIZE_VEC_COUNT = HEAD_SIZE / VEC_SIZE;
  constexpr int VEC_LOAD_Y_STEP = BLOCK_THREADS / HEAD_SIZE_VEC_COUNT;
  constexpr int Q_REG_COUNT = BLOCK_Q / VEC_LOAD_Y_STEP;

  // vectorize load q
  auto vec_ld_y = thread_idx % HEAD_SIZE_VEC_COUNT;
  auto vec_ld_x = thread_idx / HEAD_SIZE_VEC_COUNT;
  vec_t q_reg[Q_REG_COUNT];
#pragma unroll
  for (int i = 0; i < Q_REG_COUNT; i++) {
    q_reg[i] = reinterpret_cast<vec_t*>(
        q_bs + (vec_ld_y + i * VEC_LOAD_Y_STEP) * HEAD_SIZE)[vec_ld_x];
  }
#pragma unroll
  for (int i = 0; i < Q_REG_COUNT; i++) {
    auto offset =
        vec_ld_x * VEC_SIZE * BLOCK_Q + vec_ld_y + i * VEC_LOAD_Y_STEP;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      auto idx = offset + j * BLOCK_Q; // col major
      slm->q[idx] = q_reg[i].val[j];
    }
  }

  // for (int i = thread_idx; i < BLOCK_Q * HEAD_SIZE / VEC_SIZE;
  //      i += BLOCK_THREADS) {
  //   reinterpret_cast<vec_t*>(o_bs)[i] = vec_t{0};
  // }
  // item.barrier(dpcpp_local_fence);

  for (int start_s = 0; start_s < kv_seq_length; start_s += BLOCK_KV) {
    // vectorize load k, v
    constexpr int KV_REG_COUNT = BLOCK_KV / VEC_LOAD_Y_STEP;
    vec_t k_reg[KV_REG_COUNT];
    vec_t v_reg[KV_REG_COUNT];
#pragma unroll
    for (int i = 0; i < KV_REG_COUNT; i++) {
      k_reg[i] = reinterpret_cast<vec_t*>(
          k_b +
          (start_s + vec_ld_y + i * VEC_LOAD_Y_STEP) * HEAD_SIZE)[vec_ld_x];
    }
#pragma unroll
    for (int i = 0; i < KV_REG_COUNT; i++) {
      v_reg[i] = reinterpret_cast<vec_t*>(
          v_b +
          (start_s + vec_ld_y + i * VEC_LOAD_Y_STEP) * HEAD_SIZE)[vec_ld_x];
    }

#pragma unroll
    for (int i = 0; i < KV_REG_COUNT; i++) {
      auto offset =
          vec_ld_x * VEC_SIZE * BLOCK_KV + vec_ld_y + i * VEC_LOAD_Y_STEP;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        auto idx = offset + j * BLOCK_KV; // col major
        slm->k[idx] = k_reg[i].val[j];
      }
    }
    item.barrier(dpcpp_local_fence);

    group::fmha_naive_gemm<
        scalar_t,
        BLOCK_Q,
        BLOCK_KV,
        HEAD_SIZE,
        BLOCK_THREADS,
        /*A_ROW_MAJOR*/ false,
        /*B_ROW_MAJOR*/ true,
        /*C_ROW_MAJOR*/ true,
        /*ALLOW_ACC_OUT*/ false,
        /*FMHA_AUTO_BETA_AND_VEC_WR*/ false,
        /*IS_CAUSAL*/ IS_CAUSAL>(
        item,
        slm->qk,
        slm->q,
        slm->k,
        1.0f / std::sqrt((float)HEAD_SIZE),
        0.0,
        nullptr,
        nullptr,
        block_idx_1 * BLOCK_Q,
        start_s);
    item.barrier(dpcpp_local_fence);

    group::fmha_reduce_max_x_sync<scalar_t, BLOCK_Q, BLOCK_KV>(
        item, slm->m_curr, slm->qk, slm->m_prev);

    if (thread_idx < BLOCK_Q) {
      slm->l_prev[thread_idx] *=
          std::exp(slm->m_prev[thread_idx] - slm->m_curr[thread_idx]);
    }
    for (int i = thread_idx; i < BLOCK_Q * BLOCK_KV; i += BLOCK_THREADS) {
      int j = i / BLOCK_KV;
      slm->p[i] = std::exp(slm->qk[i] - slm->m_curr[j]);
    }
    item.barrier(dpcpp_local_fence);

    group::fmha_reduce_sum_x_sync<scalar_t, BLOCK_Q, BLOCK_KV>(
        item, slm->l_curr, slm->p, slm->l_prev);

    if (thread_idx < BLOCK_Q) {
      slm->l_rcp[thread_idx] = 1.0f / slm->l_curr[thread_idx];
    }
    item.barrier(dpcpp_local_fence);

#pragma unroll
    for (int i = 0; i < KV_REG_COUNT; i++) {
      auto offset =
          (vec_ld_y + i * VEC_LOAD_Y_STEP) * HEAD_SIZE + vec_ld_x * VEC_SIZE;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        slm->v[offset + j] = v_reg[i].val[j];
      }
    }
    item.barrier(dpcpp_local_fence);

    group::fmha_naive_gemm<
        scalar_t,
        BLOCK_Q,
        HEAD_SIZE,
        BLOCK_KV,
        BLOCK_THREADS,
        true,
        true,
        true,
        true,
        true,
        false>(item, o_bs, slm->p, slm->v, 1.0, 1.0, slm->l_rcp, slm->l_prev);
    item.barrier(dpcpp_local_fence);

    if (thread_idx < BLOCK_Q) {
      slm->l_prev[thread_idx] = slm->l_curr[thread_idx];
      slm->m_prev[thread_idx] = slm->m_curr[thread_idx];
    }
    item.barrier(dpcpp_local_fence);
  }

  if (thread_idx < BLOCK_Q / VEC_SIZE) {
    reinterpret_cast<vec_t*>(o_ms)[thread_idx] =
        reinterpret_cast<vec_t*>(slm->m_prev)[thread_idx];
    reinterpret_cast<vec_t*>(o_ls)[thread_idx] =
        reinterpret_cast<vec_t*>(slm->l_prev)[thread_idx];
  }
}

} // namespace impl

template <
    typename scalar_t,
    int BLOCK_Q,
    int BLOCK_KV,
    int HEAD_SIZE,
    typename queue_t>
void scaled_dot_product_attention_impl(
    queue_t& queue,
    scalar_t* out,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const int batch_size,
    const int num_heads,
    const int q_seq_length,
    const int kv_seq_length,
    const bool is_causal,
    scalar_t* out_m,
    scalar_t* out_l) {
  constexpr int BLOCK_THREADS = 256;
  constexpr int WARP_SIZE = 32;
  constexpr int VEC_SIZE = 4;

  static_assert(HEAD_SIZE % VEC_SIZE == 0);
  static_assert(BLOCK_Q % VEC_SIZE == 0);

  TORCH_CHECK(q_seq_length % BLOCK_Q == 0);
  TORCH_CHECK(kv_seq_length % BLOCK_KV == 0);

  if (batch_size * num_heads * q_seq_length * kv_seq_length == 0)
    return;
  auto slm_size = sizeof(
      impl::memory::fmha_shared_memory<scalar_t, BLOCK_Q, BLOCK_KV, HEAD_SIZE>);

  auto cgf = DPCPP_Q_CGF(h) {
    auto slm_ = sycl::local_accessor<char>(slm_size, h);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(WARP_SIZE)]] {
      auto slm_ptr = (char*)slm_.get_pointer().get();
      if (is_causal)
        impl::scaled_dot_product_attention_kernel<
            scalar_t,
            BLOCK_Q,
            BLOCK_KV,
            HEAD_SIZE,
            BLOCK_THREADS,
            VEC_SIZE,
            WARP_SIZE,
            true>(
            item,
            out,
            q,
            k,
            v,
            q_seq_length,
            kv_seq_length,
            out_m,
            out_l,
            slm_ptr);
      else
        impl::scaled_dot_product_attention_kernel<
            scalar_t,
            BLOCK_Q,
            BLOCK_KV,
            HEAD_SIZE,
            BLOCK_THREADS,
            VEC_SIZE,
            WARP_SIZE,
            false>(
            item,
            out,
            q,
            k,
            v,
            q_seq_length,
            kv_seq_length,
            out_m,
            out_l,
            slm_ptr);
    };
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(
                batch_size * num_heads * q_seq_length / BLOCK_Q *
                BLOCK_THREADS),
            sycl::range<1>(BLOCK_THREADS)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> naive_scaled_dot_product(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool is_causal) {
  TORCH_CHECK(
      q.is_xpu() && k.is_xpu() && v.is_xpu(), "inputs must be a XPU tensor");

  int batch_size = q.size(0);
  int num_heads = q.size(1);
  int q_seq_length = q.size(2);
  int kv_seq_length = k.size(2);
  int head_size = q.size(3);
  TORCH_CHECK(
      k.size(0) == batch_size && k.size(1) == num_heads &&
      k.size(3) == head_size);
  TORCH_CHECK(
      v.size(0) == batch_size && v.size(1) == num_heads &&
      v.size(3) == head_size);

  constexpr int BLOCK_Q = 64;
  constexpr int BLOCK_KV = 64;
  constexpr int HEAD_SIZE = 64;
  constexpr int BLOCK_THREADS = 256;
  constexpr int VEC_SIZE = 4;

  TORCH_CHECK(q_seq_length % BLOCK_Q == 0);
  TORCH_CHECK(kv_seq_length % BLOCK_KV == 0);
  TORCH_CHECK(head_size == HEAD_SIZE);

  auto out = at::zeros_like(q);
  auto out_m = at::empty({batch_size, num_heads, q_seq_length}, q.options());
  auto out_l = at::empty({batch_size, num_heads, q_seq_length}, q.options());

  if (batch_size * num_heads * q_seq_length * kv_seq_length == 0)
    return std::forward_as_tuple(out, out_m, out_l);

  auto& queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      q.scalar_type(), "naive_scaled_dot_product", [&] {
        auto out_ptr = (scalar_t*)out.data_ptr();
        auto q_ptr = (scalar_t*)q.data_ptr();
        auto k_ptr = (scalar_t*)k.data_ptr();
        auto v_ptr = (scalar_t*)v.data_ptr();
        auto out_m_ptr = (scalar_t*)out_m.data_ptr();
        auto out_l_ptr = (scalar_t*)out_l.data_ptr();
        scaled_dot_product_attention_impl<scalar_t, 64, 16, HEAD_SIZE>(
            queue,
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            batch_size,
            num_heads,
            q_seq_length,
            kv_seq_length,
            is_causal,
            out_m_ptr,
            out_l_ptr);
      });

  return std::forward_as_tuple(out, out_m, out_l);
}

} // namespace AtenIpexTypeXPU
} // namespace at
