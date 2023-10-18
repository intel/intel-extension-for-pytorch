
#include <ATen/record_function.h>
#include <aten/TPPGEMM.h>
#include <torch/all.h>
#include <iostream>
#include <vector>
#include "tpp/ext_tpp.h"
#include "tpp/utils.h"
#ifndef NO_PARLOOPER
#include "tpp/threaded_loops.h"
#endif
#include <cstdint>
#include "tpp/tensor_helper.h"
#include "tpp/xsmm_functors.h"

namespace torch_ipex {
namespace tpp {

static int large_cache_opt = false;
static int use_at_vnni = false; // env2int("USE_AT_VNNI");
static int FT_OPT_SIZE = env2int("FT_OPT_SIZE", 256);
static int NCB_BLOCK_SIZE = env2int("NCB_BLOCK_SIZE", 64);
static const char* GEMM_LOOP_SCHEME =
    getenv("GEMM_LOOP_SCHEME") ? getenv("GEMM_LOOP_SCHEME") : "aCB";

REGISTER_LOCAL_SCOPE(
    tpp_linear_krnl,
    "tpp_linear_krnl"); //  linear W/ and W/O bias
REGISTER_LOCAL_SCOPE(
    tpp_linear_add_add_krnl,
    "tpp_linear_add_add_krnl"); // linear bias + add + add
REGISTER_LOCAL_SCOPE(
    tpp_linear_gelu_krnl,
    "tpp_linear_gelu_krnl"); // linear bias + gelu

REGISTER_LOCAL_SCOPE(
    tpp_linear_mul_krnl,
    "tpp_linear_mul_krnl"); // linear bias + mul
REGISTER_LOCAL_SCOPE(
    tpp_linear_add_krnl,
    "tpp_linear_add_krnl"); // linear bias + add
REGISTER_LOCAL_SCOPE(
    tpp_linear_silu_krnl,
    "tpp_linear_silu_krnl"); // linear bias + silu
REGISTER_LOCAL_SCOPE(
    tpp_linear_relu_krnl,
    "tpp_linear_relu_krnl"); // linear bias + relu

REGISTER_LOCAL_SCOPE(fftkn, "fftkn");

template <typename T>
inline at::Tensor wt_tensor_for_first_token(at::Tensor& t) {
  RECORD_SCOPE(fftkn, {t});
  auto dim = t.dim();
  if (dim < 5)
    return t;
  auto sizes = t.sizes();
  constexpr long RBS = 2;
  auto K1 = sizes[0];
  if (K1 % RBS != 0)
    return t;
  auto C1 = sizes[1];
  auto C2 = sizes[2];
  auto K2 = sizes[3];
  auto C3 = sizes[4];

  auto t_new = t.new_empty({K1 / RBS, C1, C2, RBS * K2, C3});
  auto in = GetVLAPtr<T>(t, {RBS, C1, C2, K2 * C3});
  auto out = GetVLAPtr<T>(t_new, {C1, C2, RBS, K2 * C3});

  auto cpy_tpp =
      SCOPEIT(CpyTPP<T>(C2, K2 * C3, K2 * C3, RBS * K2 * C3), EW_COPY);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < K1 / RBS; i++) {
    for (int j = 0; j < C1; j++) {
      for (int k = 0; k < RBS; k++) {
        cpy_tpp(in[i][k][j][0], out[i][j][0][k]);
      }
    }
  }

  return t_new;
}

template <typename T>
inline void tpp_linear_bias(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
  auto in_sizes = t_in.sizes();
  auto wt_sizes = t_wt.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});

  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});

  auto bias = GetVLAPtr<T>(t_bias, {Hk});

  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_SCOPE(tpp_linear_krnl, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T, typename Tout = T>
inline void tpp_linear_no_bias(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;
  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto out = GetVLAPtr<Tout>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % BSb;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  auto zero_tpp = SCOPEIT(SetZeroTPP<Tout>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<Tout>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, Tout>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, Tout>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));

  {
    RECORD_SCOPE(tpp_linear_krnl, {t_in, t_wt_V});
    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto gemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    gemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              zero_tpp(out[s1][nk]);
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
          } else {
            if (nc == 0) {
              zero_tpp_rem(out[s1][nk]);
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_mul(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto mul_tpp = SCOPEIT((MulTPP<T, T>(BSb, Hk, K, K)), EW_MUL);
  auto mul_tpp_rem = SCOPEIT((MulTPP<T, T>(rem, Hk, K, K)), EW_MUL);

  {
    RECORD_SCOPE(tpp_linear_mul_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              mul_tpp_rem(in1[s1][nk], out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_add_add(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out,
    double scale) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto in2 = GetVLAPtr<T>(t_in2, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;
  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto add_tpp = SCOPEIT((AddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto add_tpp_rem = SCOPEIT((AddTPP<T, T>(rem, Hk, K, K)), EW_ADD);
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_SCOPE(tpp_linear_add_add_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp(in2[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              add_tpp_rem(out[s1][nk], in1[s1][nk], out[s1][nk]);
              sadd_tpp_rem(in2[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_gelu(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;
  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto gelu_fwd_tpp_rem = SCOPEIT(GeluFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(tpp_linear_gelu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              gelu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_add(
    at::Tensor t_in,
    at::Tensor t_in1,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out,
    float scale) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<T>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto sadd_tpp = SCOPEIT((ScaleAddTPP<T, T>(BSb, Hk, K, K)), EW_ADD);
  auto sadd_tpp_rem = SCOPEIT((ScaleAddTPP<T, T>(rem, Hk, K, K)), EW_ADD);

  {
    RECORD_SCOPE(tpp_linear_add_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto ogemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0L, BS, BSb}, {Nk}}, loop_scheme);
    ogemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp(in1[s1][nk], out[s1][nk], scale);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              sadd_tpp_rem(in1[s1][nk], out[s1][nk], scale);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_silu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto silu_fwd_tpp = SCOPEIT(SiLUFwdTPP<T>(BSb, Hk, K, K), ACT);
  auto silu_fwd_tpp_rem = SCOPEIT(SiLUFwdTPP<T>(rem, Hk, K, K), ACT);

  {
    RECORD_SCOPE(tpp_linear_silu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              silu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

template <typename T>
inline void tpp_linear_relu(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    at::Tensor t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  if (BS > FT_OPT_SIZE) { // first token compute
    t_wt = wt_tensor_for_first_token<T>(t_wt);
  }
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];

  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;

  auto t_wt_V = torch_ipex::tpp::wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

  auto in = GetVLAPtr<T>(t_in, {Nc, Hc});
  auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<T>(t_bias, {Hk});
  auto out = GetVLAPtr<T>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (large_cache_opt)
    Ncb = NCB_BLOCK_SIZE;

  bool with_bias = (t_bias.numel() > 0);
  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(BSb, Hk, K), BIAS);
  auto copy_bias_tpp_rem = SCOPEIT(CpyBiasTPP<T>(rem, Hk, K), BIAS);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(BSb, Hk, K), EW_ZERO);
  auto zero_tpp_rem = SCOPEIT(SetZeroTPP<T>(rem, Hk, K), EW_ZERO);
  auto brgemm_tpp = SCOPEITGEMM(
      (BrgemmTPP<T, T>(BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto brgemm_tpp_rem = SCOPEITGEMM(
      (BrgemmTPP<T, T>(rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)));
  auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<T>(BSb, Hk, K, K, false), ACT);
  auto relu_fwd_tpp_rem = SCOPEIT(ReLUFwdTPP<T>(rem, Hk, K, K, false), ACT);

  {
    RECORD_SCOPE(tpp_linear_relu_krnl, {t_in, t_wt_V});

    auto loop_scheme = large_cache_opt ? GEMM_LOOP_SCHEME : "aCb";
    auto igemm_loop = torch_ipex::tpp::ThreadedLoop<3>(
        {{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop(
        [&](int* ind) {
          int nc = ind[0], s1 = ind[1], nk = ind[2];
          auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
          bool is_rem = (s1 + BSb > BS);
          if (!is_rem) {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp(bias[nk], out[s1][nk]);
              } else {
                zero_tpp(out[s1][nk]);
              }
            }
            brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp(out[s1][nk], out[s1][nk]);
            }
          } else {
            if (nc == 0) {
              if (with_bias) {
                copy_bias_tpp_rem(bias[nk], out[s1][nk]);
              } else {
                zero_tpp_rem(out[s1][nk]);
              }
            }
            brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);
            brgemm_tpp.config();
            if (!(nc + Ncb < Nc)) { // last nc iter
              relu_fwd_tpp_rem(out[s1][nk], out[s1][nk]);
            }
          }
        },
        [&]() { brgemm_tpp.config(); },
        [&]() { brgemm_tpp.release(); });
  }
}

} // namespace tpp
} // namespace torch_ipex
