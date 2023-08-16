RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
int i = 0;
auto t_in = inputs[i++]; // [S1][Nc][S2][Hc]
auto t_in2 = inputs[i++]; // [S1][Nk][S2][Hk]
auto t_wt = inputs[i++]; // [Nk][Nc][Hc][Hk]
auto t_bias = inputs[i++]; // [Nk][Hk]
auto t_gamma = inputs[i++]; // [Nk][Hk]
auto t_beta = inputs[i++]; // [Nk][Hk]
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto S1 = in_sizes[0];
auto Nc = in_sizes[1];
auto S2 = in_sizes[2];
auto Hc = in_sizes[3];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

auto t_dout = t_in.new_empty({S1, Nk, S2, Hk});
auto t_out = t_dout;
if (training) {
  t_out = t_in.new_empty({S1, Nk, S2, Hk});
}

// auto t_dp_mask = at::Tensor();
auto t_dp_mask = at::empty({S1, Nk, (S2 * Hk + 15) / 16}, at::kShort);
auto t_mean = t_gamma.new_empty({S1, S2}, at::kFloat);
auto t_var = t_gamma.new_empty({S1, S2}, at::kFloat);

if (p > 0)
  t_dp_mask = at::empty({S1, Nk, (S2 * Hk + 15) / 16}, at::kShort);

DECL_VLA_PTR_PT(T, in, [Nc][S2][Hc], t_in);
DECL_VLA_PTR_PT(T, in2, [Nk][S2][Hk], t_in2);
// DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc / 2][Hk][2], t_wt_V);
DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc * Hk], t_wt_V);
DECL_VLA_PTR_PT(T, bias, [Hk], t_bias);
DECL_VLA_PTR_PT(T, gamma, [Hk], t_gamma);
DECL_VLA_PTR_PT(T, beta, [Hk], t_beta);
DECL_VLA_PTR_PT(float, mean, [S2], t_mean);
DECL_VLA_PTR_PT(float, var, [S2], t_var);
DECL_VLA_PTR_PT(T, dout, [Nk][S2][Hk], t_dout);
DECL_VLA_PTR_PT(T, out, [Nk][S2][Hk], t_out);
DECL_VLA_PTR_PT(short, dp_mask, [Nk][(S2 * Hk + 15) / 16], t_dp_mask);

auto Ncb = Nc;
if (Nc > Nk && Nc % Nk == 0) {
  Ncb = Nk;
}
// Create TPPs
auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, Hk), BIAS);
auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hk,
    Hc,
    S2* Hc,
    Hk* Hc,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Ncb)));
auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S2 * Hk, p), DROPOUT);
auto add_tpp = SCOPEIT((AddTPP<T, T>(S2 * Hk)), EW_ADD);
auto layer_norm_fwd_tpp =
    SCOPEIT(LayerNormFwdTPP<T>(Nk, S2, Hk, eps), LAYER_NORM);

{
  RECORD_SCOPE(o_gemm, {t_in, t_wt});
#if 0
  auto nThreads = omp_get_max_threads();
  for (int nc = 0; nc < Nc; nc += Ncb) {
    if (nc == Nc - Ncb) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      if (nThreads < S1) {
#pragma omp parallel for
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < Nk; nk++) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], dout[s1][nk][0]);
            }
            brgemm_tpp(in[s1][nc][0], wt_V[nk][nc], dout[s1][nk][0], Ncb);
            if (p > 0) {
              dropout_fwd_tpp(
                  dout[s1][nk][0],
                  (void*)get_rng_state(),
                  dout[s1][nk][0],
                  dp_mask[s1][nk]);
            }
            add_tpp(dout[s1][nk][0], in2[s1][nk][0], dout[s1][nk][0]);
          }
          layer_norm_fwd_tpp(
              dout[s1][0][0],
              gamma[0],
              beta[0],
              mean[s1],
              var[s1],
              out[s1][0][0]);
        }
      } else {
#pragma omp parallel for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < Nk; nk++) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], dout[s1][nk][0]);
            }
            brgemm_tpp(in[s1][nc][0], wt_V[nk][nc], dout[s1][nk][0], Ncb);
            if (p > 0) {
              dropout_fwd_tpp(
                  dout[s1][nk][0],
                  (void*)get_rng_state(),
                  dout[s1][nk][0],
                  dp_mask[s1][nk]);
            }
            add_tpp(dout[s1][nk][0], in2[s1][nk][0], dout[s1][nk][0]);
          }
        }
#pragma omp parallel for
        for (int s1 = 0; s1 < S1; s1++) {
          layer_norm_fwd_tpp(
              dout[s1][0][0],
              gamma[0],
              beta[0],
              mean[s1],
              var[s1],
              out[s1][0][0]);
        }
      }
    } else {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < Nk; nk++) {
          if (nc == 0) {
            copy_bias_tpp(bias[nk], dout[s1][nk][0]);
          }
          brgemm_tpp(in[s1][nc][0], wt_V[nk][nc], dout[s1][nk][0], Ncb);
        }
      }
    }
  }
#else
  auto ogemm_loop = ThreadedLoop<3>(
      {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{S1}, LoopSpecs{Nk}}, "acB");
  bool parallelized_on_nk = false; // ogemm_loop.is_parallel(2);
  ogemm_loop(
      [&](int* ind) {
        int nc = ind[0], s1 = ind[1], nk = ind[2];
        DECL_VLA_PTR_PT(T, bias, [Hk], t_bias);
        DECL_VLA_PTR_PT(T, dout, [Nk][S2 * Hk], t_dout);
        DECL_VLA_PTR_PT(T, in, [Nc][S2 * Hc], t_in);
        DECL_VLA_PTR_PT(T, in2, [Nk][S2 * Hk], t_in2);
        DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc * Hk], t_wt_V);
        DECL_VLA_PTR_PT(short, dp_mask, [Nk][(S2 * Hk + 15) / 16], t_dp_mask);
        DECL_VLA_PTR_PT(T, gamma, [Hk], t_gamma);
        DECL_VLA_PTR_PT(T, beta, [Hk], t_beta);
        DECL_VLA_PTR_PT(float, mean, [S2], t_mean);
        DECL_VLA_PTR_PT(float, var, [S2], t_var);
        DECL_VLA_PTR_PT(T, out, [Nk][S2 * Hk], t_out);
        if (nc == 0) {
          copy_bias_tpp(bias[nk], dout[s1][nk]);
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], dout[s1][nk], Ncb, true);
        if (!(nc + Ncb < Nc)) { // last nc iter
          // if (nc == Nc - Ncb) { // last nc iter
          if (p > 0) {
            dropout_fwd_tpp(
                dout[s1][nk],
                (void*)get_rng_state(),
                dout[s1][nk],
                dp_mask[s1][nk]);
          }
          add_tpp(dout[s1][nk], in2[s1][nk], dout[s1][nk]);
          if (!parallelized_on_nk && nk == Nk - 1) {
            layer_norm_fwd_tpp(
                dout[s1][0], gamma[0], beta[0], mean[s1], var[s1], out[s1][0]);
          }
        }
      },
      [&]() { brgemm_tpp.config(); },
      [&]() { brgemm_tpp.release(); });

  if (parallelized_on_nk) {
#pragma omp parallel for
    for (int s1 = 0; s1 < S1; s1++) {
      layer_norm_fwd_tpp(
          dout[s1][0][0], gamma[0], beta[0], mean[s1], var[s1], out[s1][0][0]);
    }
  }
#endif
}
return std::vector<at::Tensor>({t_out, t_dout, t_mean, t_var, t_dp_mask});
