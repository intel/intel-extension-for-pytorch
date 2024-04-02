RECORD_FUNCTION("bert_bwd", std::vector<c10::IValue>());
int i = 0;
auto t_grad_out = inputs[i++].contiguous(); // [S1][Nc][S2][Hc]
auto t_in = inputs[i++]; // [S1][Nc][S2][Hc]
auto t_wt = inputs[i++]; // [Nk][Nc][Hc][Hk]
auto t_gamma = inputs[i++]; // [Nk][Hk]
auto t_mean = inputs[i++]; // [Nk][Hk]
auto t_var = inputs[i++]; // [Nk][Hk]
auto t_dout = inputs[i++]; // [S1][Nk][S2][Hk]
auto t_dp_mask = inputs[i++];
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto S1 = in_sizes[0];
auto Nc = in_sizes[1];
auto S2 = in_sizes[2];
auto Hc = in_sizes[3];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

const auto grad_wt_flag =
    (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);
auto t_wt_TV = wt_tensor_for_bwd_compact(Nk, Hk, Nc, Hc, t_wt);

auto t_in_T = t_in;
if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
  t_in_T = act_tensor_trans_compact(S1, Nc, S2, Hc, t_in);
}
auto in_blk = LToPBlockAccessMapper<T>(S1, Nc);

auto t_grad_in2 = at::empty_like(t_grad_out);
at::Tensor t_grad_dout; //   = at::zeros_like(t_grad_out);
auto t_grad_in = at::empty_like(t_in);
auto t_grad_wt = at::empty_like(t_wt);
auto t_grad_bias = at::empty_like(t_gamma); // [Nk][Hk]
auto t_grad_gamma = at::empty_like(t_gamma); // [Nk][Hk]
auto t_grad_beta = at::empty_like(t_gamma); // [Nk][Hk]

if (p > 0) {
  t_grad_dout = at::empty_like(t_grad_out);
} else {
  t_grad_dout = t_grad_in2;
}
auto t_grad_dout_V = t_grad_dout;
if (t_grad_dout.dtype() == at::kBFloat16) {
  t_grad_dout_V = t_grad_out.new_empty({Nk, S1, S2 / 2, Hk, 2});
}
auto gdout_blk = LToPBlockAccessMapper<T>(S1, Nk);

DECL_VLA_PTR_PT(T, in_T, [Hc][S2], t_in_T);
DECL_VLA_PTR_PT(T, grad_in2, [Nk][S2][Hk], t_grad_in2);
DECL_VLA_PTR_PT(T, grad_in, [Nc][S2][Hc], t_grad_in);
DECL_VLA_PTR_PT(T, wt_TV, [Nk][Hk * Hc], t_wt_TV);
DECL_VLA_PTR_PT(T, grad_wt, [Nc][Hc][Hk], t_grad_wt);
DECL_VLA_PTR_PT(T, grad_bias, [Hk], t_grad_bias);
DECL_VLA_PTR_PT(T, gamma, [Hk], t_gamma);
DECL_VLA_PTR_PT(T, grad_gamma, [Hk], t_grad_gamma);
DECL_VLA_PTR_PT(T, grad_beta, [Hk], t_grad_beta);
DECL_VLA_PTR_PT(float, mean, [S2], t_mean);
DECL_VLA_PTR_PT(float, var, [S2], t_var);
DECL_VLA_PTR_PT(T, grad_dout, [Nk][S2][Hk], t_grad_dout);
DECL_VLA_PTR_PT(T, grad_dout_V, [S2 * Hk], t_grad_dout_V);
DECL_VLA_PTR_PT(T, dout, [Nk][S2][Hk], t_dout);
DECL_VLA_PTR_PT(T, grad_out, [Nk][S2][Hk], t_grad_out);
DECL_VLA_PTR_PT(short, dp_mask, [Nk][(S2 * Hk + 15) / 16], t_dp_mask);

constexpr int64_t BS = 8;
auto Nkb = Nk;
if (Nk > Nc && Nk % Nc == 0) {
  Nkb = Nc;
}

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(Nk * Hk), EW_ZERO);
auto layer_norm_bwd_tpp = SCOPEIT(LayerNormBwdTPP<T>(Nk, S2, Hk), LAYER_NORM);
auto drop_out_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(S2 * Hk, p), DROPOUT);
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, Hk), BIAS);
auto n2v_tpp =
    SCOPEIT(XformExtTPP<T>(S2, Hk, XformTPP::XFORM_N2V_TPP, true), VNNI);
auto di_gemm_b0_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hc,
    Hk,
    S2* Hk,
    Hk* Hc,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hc,
    Hk,
    S2* Hk,
    Hk* Hc,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto dw_set_zero_tpp = SCOPEIT(SetZeroTPP<T>(Hk * Hc), EW_ZERO);
auto dw_cpy_tpp = SCOPEIT(CpyTPP<T>(Hk * Hc), VNNI);
auto dw_n2v_tpp =
    SCOPEIT(XformExtTPP<T>(Hc, Hk, XformTPP::XFORM_N2V_TPP, true), VNNI);
auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    Hc,
    Hk,
    S2,
    input_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * Hc : Nc * S2 * Hc,
    input_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * Hk : Nk * S2 * Hk,
    1.0,
    XformTPP::XFORM_NONE_TPP, //(XformTPP::XFORM_TYPE)grad_wt_flag,
    input_trans_flag,
    BS)));

{
  RECORD_SCOPE(do_bias, {t_grad_out});
  int num_threads = omp_get_max_threads();
  float* gamma_ptrs[num_threads];
  float* beta_ptrs[num_threads];
  float* bias_ptrs[num_threads];
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[Nk][Hk];
      float prv_grad_gamma[Nk][Hk];
      float prv_grad_beta[Nk][Hk];
      bias_ptrs[tid] = prv_grad_bias[0];
      beta_ptrs[tid] = prv_grad_beta[0];
      gamma_ptrs[tid] = prv_grad_gamma[0];
      set_zero_tpp(prv_grad_bias[0]);
      set_zero_tpp(prv_grad_gamma[0]);
      set_zero_tpp(prv_grad_beta[0]);
#pragma omp for
      for (int s1 = 0; s1 < S1; s1++) {
        layer_norm_bwd_tpp(
            grad_out[s1][0][0],
            dout[s1][0][0],
            mean[s1],
            var[s1],
            gamma[0],
            grad_in2[s1][0][0],
            prv_grad_gamma[0],
            prv_grad_beta[0]);
        for (int nk = 0; nk < Nk; nk++) {
          if (p > 0) {
            drop_out_bwd_tpp(
                grad_in2[s1][nk][0], grad_dout[s1][nk][0], dp_mask[s1][nk]);
          }
          grad_bias_tpp(grad_dout[s1][nk][0], prv_grad_bias[nk]);
          n2v_tpp(grad_dout[s1][nk][0], grad_dout_V[gdout_blk(s1, nk)]);
        }
      }
      omp_reduce_buf(num_threads, Nk * Hk, gamma_ptrs, grad_gamma[0]);
      omp_reduce_buf(num_threads, Nk * Hk, beta_ptrs, grad_beta[0]);
      omp_reduce_buf(num_threads, Nk * Hk, bias_ptrs, grad_bias[0]);
    }
  }
}
{
  RECORD_SCOPE(dio_gemm, {t_grad_dout, t_wt_TV});
#if 0
  for (int nk = 0; nk < Nk; nk += Nkb) {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int s1 = 0; s1 < S1; s1++) {
      for (int nc = 0; nc < Nc; nc++) {
        if (nk == 0)
          di_gemm_b0_tpp(
              grad_dout[s1][nk][0], wt_TV[nc][nk], grad_in[s1][nc][0], Nkb);
        else
          di_gemm_b1_tpp(
              grad_dout[s1][nk][0], wt_TV[nc][nk], grad_in[s1][nc][0], Nkb);
      }
    }
  }
#else
  auto di_loop = ThreadedLoop<3>(
      {LoopSpecs{0, Nk, Nkb, false}, LoopSpecs{S1}, LoopSpecs{Nc}}, "acB");
  di_loop(
      [&](int* ind) {
        int nk = ind[0], s1 = ind[1], nc = ind[2];
        DECL_VLA_PTR_PT(T, grad_dout, [Nk][S2][Hk], t_grad_dout);
        DECL_VLA_PTR_PT(T, wt_TV, [Nk][Hk * Hc], t_wt_TV);
        DECL_VLA_PTR_PT(T, grad_in, [Nc][S2][Hc], t_grad_in);
        if (nk == 0)
          di_gemm_b0_tpp(
              grad_dout[s1][nk][0],
              wt_TV[nc][nk],
              grad_in[s1][nc][0],
              Nkb,
              true);
        else
          di_gemm_b1_tpp(
              grad_dout[s1][nk][0],
              wt_TV[nc][nk],
              grad_in[s1][nc][0],
              Nkb,
              true);
      },
      [&]() { di_gemm_b0_tpp.config(); },
      [&]() { di_gemm_b0_tpp.release(); });
#endif
}
{
  RECORD_SCOPE(dwo_gemm, {t_in_T, t_grad_dout_V});
#if 0
  for (int s1 = 0; s1 < S1; s1 += BS) {
    int count = (s1 + BS <= S1 ? BS : S1 - s1);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int nk = 0; nk < Nk; nk++) {
      for (int nc = 0; nc < Nc; nc++) {
        if (s1 == 0)
          dw_set_zero_tpp(grad_wt[nk][nc]);
        dw_gemm_tpp(in_T[in_blk(s1, nc)], grad_dout_V[gdout_blk(s1, nk)], grad_wt[nk][nc], count);
        bool is_last_iter = !(s1 + BS < S1);
        if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
          T tmp[Hc * Hk];
          dw_cpy_tpp(grad_wt[nk][nc], tmp);
          dw_n2v_tpp(tmp, grad_wt[nk][nc]);
        }
      }
    }
  }
#else
  auto dw_loop = ThreadedLoop<3>(
      {LoopSpecs{0, S1, BS, true}, LoopSpecs{Nc}, LoopSpecs{Nk}}, "aBC");
  dw_loop(
      [&](int* ind) {
        int s1 = ind[0], nc = ind[1], nk = ind[2];
        int count = (s1 + BS <= S1 ? BS : S1 - s1);
        DECL_VLA_PTR_PT(T, grad_wt, [Nc][Hc * Hk], t_grad_wt);
        DECL_VLA_PTR_PT(T, in_T, [Hc * S2], t_in_T);
        DECL_VLA_PTR_PT(T, grad_dout_V, [S2 * Hk], t_grad_dout_V);
        if (s1 == 0)
          dw_set_zero_tpp(grad_wt[nk][nc]);
#if 1
        dw_gemm_tpp(
            in_T[in_blk(s1, nc)],
            grad_dout_V[gdout_blk(s1, nk)],
            grad_wt[nk][nc],
            count,
            true);
        bool is_last_iter = !(s1 + BS < S1);
        if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
          T tmp[Hc * Hk];
          dw_cpy_tpp(grad_wt[nk][nc], tmp);
          dw_n2v_tpp(tmp, grad_wt[nk][nc]);
        }
#else
        bool is_last_iter = !(s1 + BS < S1);
        if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
          T tmp[Hc * Hk];
          dw_cpy_tpp(grad_wt[nk][nc], tmp);
          dw_gemm_tpp(
              in_T[in_blk(s1, nc)],
              grad_dout_V[gdout_blk(s1, nk)],
              tmp,
              count,
              true);
          dw_n2v_tpp(tmp, grad_wt[nk][nc]);
        } else {
          dw_gemm_tpp(
              in_T[in_blk(s1, nc)],
              grad_dout_V[gdout_blk(s1, nk)],
              grad_wt[nk][nc],
              count,
              true);
        }
#endif
      },
      [&]() { dw_gemm_tpp.config(); },
      [&]() { dw_gemm_tpp.release(); });
#endif
}
return std::vector<at::Tensor>(
    {t_grad_in, t_grad_in2, t_grad_wt, t_grad_bias, t_grad_gamma, t_grad_beta});
