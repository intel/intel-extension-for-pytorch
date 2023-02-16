RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto S1 = in_sizes[0];
auto Nc = in_sizes[1];
auto S2 = in_sizes[2];
auto Hc = in_sizes[3];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

auto t_gelu_out = t_in.new_empty({S1, Nk, S2, Hk});
auto t_out = t_gelu_out;
if (training) {
  t_out = t_in.new_empty({S1, Nk, S2, Hk});
}

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
auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(S2 * Hk), ACT);

{
  RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
#if 0
DECL_VLA_PTR_PT(T, in, [Nc][S2 * Hc], t_in);
DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc * Hk], t_wt_V);
DECL_VLA_PTR_PT(T, bias, [Hk], t_bias);
DECL_VLA_PTR_PT(T, out, [Nk][S2 * Hk], t_out);
DECL_VLA_PTR_PT(T, gelu_out, [Nk][S2 * Hk], t_gelu_out);
  for (int nc = 0; nc < Nc; nc += Ncb) {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int s1 = 0; s1 < S1; s1++) {
      for (int nk = 0; nk < Nk; nk++) {
        if (nc == 0) {
          copy_bias_tpp(bias[nk], out[s1][nk]);
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], Ncb);
        if (nc == Nc - Ncb) { // last iter
          gelu_fwd_tpp(out[s1][nk], gelu_out[s1][nk]);
        }
      }
    }
  }
#else
  auto gemm_loop = ThreadedLoop<3>(
      {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{S1}, LoopSpecs{Nk}}, "acB");
  gemm_loop(
      [&](int* ind) {
        int nc = ind[0], s1 = ind[1], nk = ind[2];
        DECL_VLA_PTR_PT(T, in, [Nc][S2 * Hc], t_in);
        DECL_VLA_PTR_PT(T, wt_V, [Nc][Hc * Hk], t_wt_V);
        DECL_VLA_PTR_PT(T, bias, [Hk], t_bias);
        DECL_VLA_PTR_PT(T, out, [Nk][S2 * Hk], t_out);
        DECL_VLA_PTR_PT(T, gelu_out, [Nk][S2 * Hk], t_gelu_out);

        if (nc == 0) {
          copy_bias_tpp(bias[nk], out[s1][nk]);
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], Ncb, true);
        if (nc == Nc - Ncb) { // last iter
          gelu_fwd_tpp(out[s1][nk], gelu_out[s1][nk]);
        }
      },
      [&]() { brgemm_tpp.config(); },
      [&]() { brgemm_tpp.release(); });

#endif
}
// if (at::isnan(t_out).any().item<bool>()) std::cout << "t_out has NaN" <<
// std::endl; if (at::isnan(t_gelu_out).any().item<bool>()) std::cout <<
// "t_gelu_out has NaN" << std::endl;
return std::vector<at::Tensor>({t_out, t_gelu_out});
