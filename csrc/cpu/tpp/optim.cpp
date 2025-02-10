//#include "init.h"
#ifdef ENABLE_RTM
#include "rtm.h"
#endif
#include "timing.h"
#include "xsmm_functors.h"

#if (defined(__x86_64__) || defined(__i386__))
#include "ATen/native/cpu/Intrinsics.h"
#else
#define _mm_pause()
#endif

#include <atomic>

namespace torch_ipex {
namespace tpp {

static inline void atomic_add_float(double* dst, double fvalue) {
  typedef union {
    unsigned long long intV;
    double floatV;
  } uf64_t;

  uf64_t new_value, old_value;
  std::atomic<unsigned long long>* dst_intV =
      (std::atomic<unsigned long long>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned long long* old_intV = (unsigned long long*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(
      dst_intV, old_intV, new_value.intV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

static inline void atomic_add_float(float* dst, float fvalue) {
  typedef union {
    unsigned intV;
    float floatV;
  } uf32_t;

  uf32_t new_value, old_value;
  std::atomic<unsigned>* dst_intV = (std::atomic<unsigned>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned* old_intV = (unsigned*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(
      dst_intV, old_intV, new_value.intV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

#define MYASSERT(x)                     \
  do {                                  \
    if (!(x)) {                         \
      printf("Assert failed %s\n", #x); \
      exit(1);                          \
    }                                   \
  } while (0)

REGISTER_SCOPE(split_sgd_sparse, "splitsgd_s");
REGISTER_SCOPE(split_sgd_dense, "splitsgd_d");
REGISTER_SCOPE(dense_sparse_add, "sprse_add");
REGISTER_SCOPE(fused_adamw, "fused_adamw");
REGISTER_SCOPE(fused_lamb, "fused_lamb");
REGISTER_SCOPE(splt_adamw, "splt_adamw");
REGISTER_SCOPE(splt_lamb, "splt_lamb");
REGISTER_SCOPE(grad_norm, "grad_norm");

static int sparse_add_use_lock_free() {
  static int lock_free = -1;
  if (lock_free != -1)
    return lock_free;
#ifdef ENABLE_RTM
  char* str = getenv("PCL_USE_RTM_UPDATE");
#else
  char* str = NULL;
#endif
  if (str && atoi(str) > 0) {
    lock_free = 0;
    printf("PCL_SPARSE_ADD: Using RTM Based Update\n");
  } else {
    lock_free = 1;
    printf("PCL_SPARSE_ADD: Using Lock Free Update\n");
  }
  return lock_free;
}

template <typename scalar_t>
void dense_sparse_add_tmpl(
    at::Tensor t_dense,
    at::Tensor t_sparse,
    float alpha) {
  auto NS = t_sparse._nnz();
  auto M = t_dense.size(0);
  auto E = t_dense.size(1);
  auto t_values = t_sparse._values();
  auto t_indices = t_sparse._indices();

  PCL_ASSERT(t_dense.is_contiguous(), "dense tensor must be contiguous\n");
  // Not using below due to spurious compiler warnings
  // DECL_VLA_PTR_PT(scalar_t, dense, [E], t_dense);
  // DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  auto dense = t_dense.data_ptr<scalar_t>();
  auto values = t_values.data_ptr<scalar_t>();
  auto indices = t_indices.data_ptr<long>();
  auto lr = alpha;

  auto embbag_upd = ScaleAddTPP<scalar_t, scalar_t>(E);

  int max_thr = omp_get_max_threads();
  int use_lock_free = sparse_add_use_lock_free();
  if (use_lock_free) {
    int nthr = max_thr;
    if (M < nthr)
      nthr = M;
#pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      long j_begin = (tid * M) / nthr;
      long j_end = ((tid + 1) * M) / nthr;
      for (long i = 0; i < NS; i++) {
        auto ind = indices[i];
        if (ind >= j_begin && ind < j_end) {
          auto wa = &dense[ind * E];
          auto va = &values[i * E];
          embbag_upd(va, wa, lr);
        }
      }
    }
  } else {
#ifdef ENABLE_RTM
    SimpleSpinLock fallBackLock;
#pragma omp parallel for
    for (int i = 0; i < NS; i++) {
      auto ind = indices[i];
      auto wa = &dense[ind * E];
      auto va = &values[i * E];
      {
        TransactionScope guard(fallBackLock, 100);
        embbag_upd(va, wa, lr);
      }
    }
#else
    printf("Please compile with ENABLE_RTM set\n");
    exit(1);
#endif
  }
}

void dense_sparse_add_(
    at::Tensor dense,
    at::Tensor sparse,
    /*torch::Scalar*/ float alpha) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(dense_sparse_add, {dense, sparse, alpha});
  if (dense.dtype() == at::kFloat) {
    dense_sparse_add_tmpl<float>(dense, sparse, alpha);
    //} else if (dense.dtype() == at::kBFloat16) {
    //  dense_sparse_add_tmpl<bfloat16>(dense, sparse, alpha);
    //} else if (dense.dtype() == at::kHalf) {
    //  dense_sparse_add_tmpl<half>(dense, sparse, alpha);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }
}

void bf16_split_add_(
    at::Tensor hi_bits,
    at::Tensor lo_bits,
    at::Tensor grad,
    float lr) {
  GlobalPass _gp(UPD);
  MYASSERT(hi_bits.is_contiguous() && lo_bits.is_contiguous());
  grad = grad.contiguous();
  if (grad.is_sparse()) {
    RECORD_SCOPE(split_sgd_sparse, {hi_bits});
    auto sparse = grad;
    auto NS = sparse._nnz();
    auto M = hi_bits.size(0);
    auto E = hi_bits.size(1);
    auto values_tensor = sparse._values();
    auto indices = sparse._indices();
    auto indices_data = indices.data_ptr<long>();
    auto split_sgd_kernel = SplitSGDTPP(E);

    auto hi_data = (unsigned short*)hi_bits.data_ptr();
    auto lo_data = (unsigned short*)lo_bits.data_ptr();
    auto values_data = values_tensor.data_ptr<at::BFloat16>();
    int max_thr = omp_get_max_threads();
    int use_lock_free = sparse_add_use_lock_free();
    if (use_lock_free) {
      int nthr = max_thr;
      if (M < nthr)
        nthr = M;
#pragma omp parallel num_threads(nthr)
      {
        int tid = omp_get_thread_num();
        long j_begin = (tid * M) / nthr;
        long j_end = ((tid + 1) * M) / nthr;
        for (long i = 0; i < NS; i++) {
          auto ind = indices_data[i];
          if (ind >= j_begin && ind < j_end) {
            auto ha = &hi_data[ind * E];
            auto la = &lo_data[ind * E];
            auto va = &values_data[i * E];
            split_sgd_kernel((at::BFloat16*)ha, (at::BFloat16*)la, va, lr);
          }
        }
      }
    } else {
#ifdef ENABLE_RTM
      SimpleSpinLock fallBackLock;
#pragma omp parallel for
      for (long i = 0; i < NS; i++) {
        auto ind = indices_data[i];
        auto ha = &hi_data[ind * E];
        auto la = &lo_data[ind * E];
        auto va = &values_data[i * E];
        {
          TransactionScope guard(fallBackLock, 100);
          split_sgd_kernel((at::BFloat16*)ha, (at::BFloat16*)la, va, lr);
        }
      }
#else
      printf("Please compile with ENABLE_RTM set\n");
      exit(1);
#endif
    }
  } else {
    RECORD_SCOPE(split_sgd_dense, {hi_bits});
    auto hi_ptr = (unsigned short*)hi_bits.data_ptr();
    auto lo_ptr = (unsigned short*)lo_bits.data_ptr();
    auto grad_ptr = grad.data_ptr<at::BFloat16>();
    long sz = hi_bits.numel();
    constexpr int block_size = 64;
    auto split_sgd_kernel = SplitSGDTPP(block_size);
    long i = 0;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, block_size); i += block_size) {
      split_sgd_kernel(
          (at::BFloat16*)(hi_ptr + i),
          (at::BFloat16*)(lo_ptr + i),
          grad_ptr + i,
          lr);
    }
    if (i < sz) {
      auto split_sgd_kernel = SplitSGDTPP(sz - i);
      split_sgd_kernel(
          (at::BFloat16*)(hi_ptr + i),
          (at::BFloat16*)(lo_ptr + i),
          grad_ptr + i,
          lr);
    }
  }
}

void fused_adamw(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_adamw, {t_data});
  typedef float T;
  auto data = t_data.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data.numel();
  constexpr int BS = 64;

  auto adamw_tpp =
      SCOPEIT(FusedAdamWTPP<T>(BS, beta1, beta2, weight_decay, eps), OPTIM);

  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
  if (i < sz) {
    auto adamw_tpp = SCOPEIT(
        FusedAdamWTPP<T>(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
}

void fused_split_adamw(
    at::Tensor& t_data_hi,
    at::Tensor& t_data_lo,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(splt_adamw, {t_data_hi});
  typedef bfloat16 T;
  auto data_hi = t_data_hi.data_ptr<T>();
  auto data_lo = t_data_lo.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data_hi.numel();
  constexpr int BS = 64;

  auto split_adamw_tpp =
      SCOPEIT(FusedSplitAdamWTPP(BS, beta1, beta2, weight_decay, eps), OPTIM);

  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    split_adamw_tpp(
        &data_hi[i],
        &data_lo[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        step_size,
        lr);
  }
  if (i < sz) {
    auto split_adamw_tpp = SCOPEIT(
        FusedSplitAdamWTPP(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    split_adamw_tpp(
        &data_hi[i],
        &data_lo[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        step_size,
        lr);
  }
}

template <typename T>
double norm2(T* ptr, long N) {
  constexpr int BS = 256;
  auto norm_tpp = SCOPEIT((Norm2TPP<T, double>(BS)), OPTIM);
  double sum = 0.0f;
  long i;
#pragma omp parallel for reduction(+ : sum) lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    norm_tpp(&ptr[i], &sum);
  }
  if (i < N) {
    auto norm_tpp = SCOPEIT((Norm2TPP<T, double>(N - i)), OPTIM);
    norm_tpp(&ptr[i], &sum);
  }
  return sum;
}

template <typename T>
void tensor_scale(T* ptr, long N, float scale) {
  constexpr int BS = 256;
  auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(BS)), EW_SCL);
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
  if (i < N) {
    auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(N - i)), EW_SCL);
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
}

double clip_grad_norm(std::vector<at::Tensor>& grads, double max_norm) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(grad_norm);
  double total_norm = 0.0;
  int N = grads.size();

  for (int i = 0; i < N; i++) {
    if (grads[i].dtype() == at::kFloat) {
      total_norm += norm2(grads[i].data_ptr<float>(), grads[i].numel());
    } else if (grads[i].dtype() == at::kBFloat16) {
      total_norm += norm2(grads[i].data_ptr<bfloat16>(), grads[i].numel());
    } else {
      PCL_ASSERT(0, "Unsupported data type");
    }
  }

  total_norm = sqrt(total_norm);
  float clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1.0) {
    for (int i = 0; i < N; i++) {
      if (grads[i].dtype() == at::kFloat) {
        tensor_scale(grads[i].data_ptr<float>(), grads[i].numel(), clip_coef);
      } else if (grads[i].dtype() == at::kBFloat16) {
        tensor_scale(
            grads[i].data_ptr<bfloat16>(), grads[i].numel(), clip_coef);
      } else {
        PCL_ASSERT(0, "Unsupported data type");
      }
    }
  }
  // printf("total_norm = %g\n", total_norm);
  return total_norm;
}

float fused_lamb(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float weight_norm,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_lamb, {t_data});
  typedef float T;
  auto t_adam_step = at::empty_like(t_data);
  auto data = t_data.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  auto adam_step = t_adam_step.data_ptr<T>();
  long sz = t_data.numel();
  constexpr int BS = 64;

  auto adam_step_tpp = SCOPEIT(
      FusedAdamStepTPP<T>(BS, beta1, beta2, eps, weight_decay > 0.0, false),
      OPTIM);
  auto norm_tpp = SCOPEIT(Norm2TPP<T>(BS), OPTIM);
  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);

  long i;
  float adam_norm = 0.0f;
#pragma omp parallel for lastprivate(i) reduction(+ : adam_norm)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    adam_step_tpp(
        &data[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        &adam_step[i],
        weight_decay);
    norm_tpp(&adam_step[i], &adam_norm);
  }
  if (i < sz) {
    auto adam_step_tpp = SCOPEIT(
        FusedAdamStepTPP<T>(
            sz - i, beta1, beta2, eps, weight_decay > 0.0, false),
        OPTIM);
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(sz - i), OPTIM);
    adam_step_tpp(
        &data[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        &adam_step[i],
        weight_decay);
    norm_tpp(&adam_step[i], &adam_norm);
  }

  adam_norm = sqrtf(adam_norm);
  if (weight_norm == -1.0) {
    weight_norm = sqrtf(norm2(data, sz));
  }

  auto trust_ratio = 1.0;
  if (weight_norm != 0 && adam_norm != 0) {
    trust_ratio = weight_norm / adam_norm;
  }

  lr = -lr * trust_ratio;

  float new_weight_norm = 0.0;

#pragma omp parallel for lastprivate(i) reduction(+ : new_weight_norm)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    scale_add_tpp(&adam_step[i], &data[i], lr);
    norm_tpp(&data[i], &new_weight_norm);
  }
  if (i < sz) {
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(sz - i), OPTIM);
    auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(sz - i)), OPTIM);
    scale_add_tpp(&adam_step[i], &data[i], lr);
    norm_tpp(&data[i], &new_weight_norm);
  }
  new_weight_norm = sqrtf(new_weight_norm);
  if (new_weight_norm > 10.0)
    new_weight_norm = 10.0;
  return new_weight_norm;
}

template <typename T, typename TN>
void fused_lamb_v2_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    at::Tensor& t_adam_step,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    at::Tensor& t_weight_norms,
    at::Tensor& t_update_norms,
    float weight_decay,
    float beta1,
    float beta2,
    float lr,
    float eps,
    int block_size,
    int step,
    bool fused_param_norm) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  DECL_VLA_PTR_PT(T, d, [BS], t_data);
  DECL_VLA_PTR_PT(T, g, [BS], t_grad);
  DECL_VLA_PTR_PT(T, m, [BS], t_exp_avg);
  DECL_VLA_PTR_PT(T, v, [BS], t_exp_avg_sq);
  DECL_VLA_PTR_PT(T, u, [BS], t_adam_step);
  DECL_VLA_PTR_PT(T, dl, [BS], t_data_low);
  // auto sz = t_block_sizes.data_ptr<int>();
  auto b2p = t_block2param.data_ptr<int>();
  auto wnorm = t_weight_norms.data_ptr<TN>();
  auto unorm = t_update_norms.data_ptr<TN>();

  auto adam_step_nwd_tpp =
      SCOPEIT(FusedAdamStepTPP<T>(BS, beta1, beta2, eps, false, true), OPTIM);
  auto adam_step_wwd_tpp =
      SCOPEIT(FusedAdamStepTPP<T>(BS, beta1, beta2, eps, true, true), OPTIM);
  auto norm_tpp = SCOPEIT((Norm2TPP<T, TN>(BS)), OPTIM);
  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);
  auto scale_add_split_bf16_tpp = SCOPEIT((SplitSGDTPP(BS)), OPTIM);

  long i;
  float b1_scale = 1.0 / (1.0 - pow(beta1, step));
  float b2_scale = 1.0 / (1.0 - pow(beta2, step));
  if (!fused_param_norm) {
    t_weight_norms.zero_();
    t_update_norms.zero_();
  }
  TN fused_adam_norm = 0.0;
  TN fused_weight_norm = 0.0;
#pragma omp parallel for reduction(+ : fused_adam_norm, fused_weight_norm)
  for (i = 0; i < num_blocks; i++) {
    TN adam_norm = 0.0f;
    TN wt_norm = 0.0f;
    int p_i = b2p[i] + 1;
    float wd = weight_decay;
    bool use_wd = (wd > 0.0);
    if (use_wd) {
      adam_step_wwd_tpp(d[i], g[i], m[i], v[i], u[i], wd, b1_scale, b2_scale);
      norm_tpp(d[i], &wt_norm);
      norm_tpp(u[i], &adam_norm);
      if (!fused_param_norm) {
        atomic_add_float(&wnorm[p_i], wt_norm);
        atomic_add_float(&unorm[p_i], adam_norm);
      }
      fused_adam_norm += adam_norm;
      fused_weight_norm += wt_norm;
    } else {
      adam_step_nwd_tpp(d[i], g[i], m[i], v[i], u[i], wd, b1_scale, b2_scale);
    }
  }
  if (weight_decay > 0.0) {
    wnorm[0] = fused_weight_norm;
    unorm[0] = fused_adam_norm;
  }

#pragma omp parallel for
  for (i = 0; i < num_blocks; i++) {
    auto trust_ratio = 1.0;
    int p_i = b2p[i] + 1;
    float wd = weight_decay;
    bool use_wd = (wd > 0.0);
    if (use_wd) {
      float weight_norm = fused_weight_norm;
      float adam_norm = fused_adam_norm;
      if (!fused_param_norm) {
        weight_norm = wnorm[p_i];
        adam_norm = unorm[p_i];
      }
      adam_norm = sqrtf(adam_norm);
      weight_norm = sqrtf(weight_norm);
      if (weight_norm != 0 && adam_norm != 0) {
        trust_ratio = weight_norm / adam_norm;
      }
    }
    float final_lr = -lr * trust_ratio;
    if (std::is_same<T, float>::value) {
      scale_add_tpp(u[i], d[i], final_lr);
    } else {
      scale_add_split_bf16_tpp(
          (at::BFloat16*)d[i],
          (at::BFloat16*)dl[i],
          (at::BFloat16*)u[i],
          final_lr);
    }
  }
}

void fused_lamb_v2(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    at::Tensor& t_adam_step,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    at::Tensor& t_weight_norms,
    at::Tensor& t_update_norms,
    float weight_decay,
    float beta1,
    float beta2,
    float lr,
    float eps,
    int block_size,
    int step,
    bool fused_param_norm) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_lamb, {t_data});

  if (t_weight_norms.dtype() == at::kFloat) {
    if (t_data.dtype() == at::kFloat) {
      fused_lamb_v2_impl<float, float>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_lamb_v2_impl<bfloat16, float>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else {
      PCL_ASSERT(0, "Should not come here\n");
    }
  } else if (t_weight_norms.dtype() == at::kDouble) {
    if (t_data.dtype() == at::kFloat) {
      fused_lamb_v2_impl<float, double>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_lamb_v2_impl<bfloat16, double>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else {
      PCL_ASSERT(0, "Should not come here\n");
    }
  } else {
    PCL_ASSERT(0, "Should not come here\n");
  }
}

} // namespace tpp
} // namespace torch_ipex

/*REGISTER_SUBMODULE(_optim, m) {
  m.def("dense_sparse_add_", &dense_sparse_add_, "Pcl pcl_dense_sparse_add");
  m.def("bf16_split_add_", &bf16_split_add_, "Pcl pcl_bf16_update");
  m.def("fused_adamw", &fused_adamw, "Fused AdamW optimizer");
  m.def(
      "fused_split_adamw",
      &fused_split_adamw,
      "Fused AdamW optimizer for BF16");
  m.def("clip_grad_norm", &clip_grad_norm, "Pcl BERT clip_grad_norm");
  m.def("fused_lamb", &fused_lamb, "Fused LAMB optimizer");
  m.def("fused_lamb_v2", &fused_lamb_v2, "Fused LAMB optimizer version 2");
}*/
