RECORD_FUNCTION("bert_bwd", std::vector<c10::IValue>());
int i = 0;
auto t_dCL = inputs[i++];
auto t_dAPO = inputs[i++];
auto t_Wq = inputs[i++]; // [HS][NH]
auto t_Wk = inputs[i++]; // [HS][NH]
auto t_Wv = inputs[i++]; // [HS][NH]
auto t_HS_T = inputs[i++]; // [B][S][HS]
auto t_HM = inputs[i++]; // Optional [B][N][S][S]
auto t_EHS_T = inputs[i++]; // [B][S][HS]
auto t_QL_T = inputs[i++];
auto t_KL_V = inputs[i++];
auto t_VL_TV = inputs[i++];
auto t_AP = inputs[i++];
auto t_APD_T = inputs[i++];
auto t_APD_mask = inputs[i++];
auto t_offs = inputs[i++]; // [B+1]
auto t_offs2 = inputs[i++]; // [B+1]

int64_t B = t_offs.sizes()[0] - 1;
int64_t SS1 = t_offs2[B].item().to<int64_t>();
auto sizes = t_dCL.sizes();
auto S1 = sizes[0];
int64_t N = sizes[1];
auto S2 = sizes[2];
int64_t H = sizes[3];
// int64_t NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
const bool S2_eq_H = (S2 == H);
constexpr int64_t BS = 8;
bool dt_bf16 = (t_dCL.dtype() == at::kBFloat16);

auto t_dQL = t_QL_T.new_empty({S1, N, S2, H});
auto t_dQL_V = t_dQL;
auto t_dKL = t_KL_V.new_empty({S1, N, S2, H});
auto t_dKL_V = t_dKL;
auto t_dVL = t_VL_TV.new_empty({S1, N, S2, H});
auto t_dVL_V = t_dVL;

auto t_dWq = t_QL_T.new_empty({N, N, H, H});
auto t_dWk = t_QL_T.new_empty({N, N, H, H});
auto t_dWv = t_QL_T.new_empty({N, N, H, H});

auto t_dBq = t_QL_T.new_empty({N * H});
auto t_dBk = t_QL_T.new_empty({N * H});
auto t_dBv = t_QL_T.new_empty({N * H});

auto t_dHS = t_QL_T.new_empty({S1, N, S2, H});
// auto t_dEHS = t_QL.new_empty({S1, N, S2, H});
at::Tensor t_dEHS; // = t_QL.new_empty({S1, N, S2, H});

auto t_dAPD = at::empty_like(t_AP);
// auto t_dAPD_V = at::empty_like(t_dAPO);
auto t_dAPD_V = t_AP.new_empty({N, SS1, S2, S2});

auto null_EHS = false;

if (t_EHS_T.numel() == 0) {
  null_EHS = true;
  t_EHS_T = t_HS_T;
  t_dEHS = t_dHS;
} else {
  t_dEHS = t_QL_T.new_empty({S1, N, S2, H});
}

auto t_dCL_V = t_dCL;
if (dt_bf16) {
  t_dQL_V = t_QL_T.new_empty({N, S1, S2 / 2, H, 2});
  t_dKL_V = t_KL_V.new_empty({N, S1, S2 / 2, H, 2});
  t_dVL_V = t_VL_TV.new_empty({N, S1, S2 / 2, H, 2});
  t_dCL_V = act_tensor_n2v_compact(S1, N, S2, H, t_dCL);
}
auto atrans_blk = LToPBlockAccessMapper<T>(S1, N);
const auto grad_wt_flag =
    (t_Wq.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto a_trans_flag =
    (dt_bf16 ? XformTPP::XFORM_NONE_TPP : XformTPP::XFORM_XPOSE_TPP);
if (grad_wt_flag == XformTPP::XFORM_N2V_TPP) {
  t_dWq = t_dWq.view({N, N, H / 2, H, 2});
  t_dWk = t_dWk.view({N, N, H / 2, H, 2});
  t_dWv = t_dWv.view({N, N, H / 2, H, 2});
  t_dAPD_V = t_dAPD_V.view({N, SS1, S2 / 2, S2, 2});
}
auto t_Wq_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wq);
auto t_Wk_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wk);
auto t_Wv_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wv);

{
  DECL_VLA_PTR_PT(T, Wq_TV, [N][H * H], t_Wq_TV);
  DECL_VLA_PTR_PT(T, Wk_TV, [N][H * H], t_Wk_TV);
  DECL_VLA_PTR_PT(T, Wv_TV, [N][H * H], t_Wv_TV);
  DECL_VLA_PTR_PT(T, dWq, [N][H * H], t_dWq);
  DECL_VLA_PTR_PT(T, dWk, [N][H * H], t_dWk);
  DECL_VLA_PTR_PT(T, dWv, [N][H * H], t_dWv);
  DECL_VLA_PTR_PT(T, dBq, [H], t_dBq);
  DECL_VLA_PTR_PT(T, dBk, [H], t_dBk);
  DECL_VLA_PTR_PT(T, dBv, [H], t_dBv);
  DECL_VLA_PTR_PT(T, QL_T, [H * S2], t_QL_T);
  DECL_VLA_PTR_PT(T, KL_V, [N][S2 * H], t_KL_V);
  DECL_VLA_PTR_PT(T, VL_TV, [N][H * S2], t_VL_TV);
  DECL_VLA_PTR_PT(T, dQL, [N][S2 * H], t_dQL);
  DECL_VLA_PTR_PT(T, dQL_V, [S2 * H], t_dQL_V);
  DECL_VLA_PTR_PT(T, dKL, [N][S2 * H], t_dKL);
  DECL_VLA_PTR_PT(T, dKL_V, [S2 * H], t_dKL_V);
  DECL_VLA_PTR_PT(T, dVL, [N][S2 * H], t_dVL);
  DECL_VLA_PTR_PT(T, dVL_V, [S2 * H], t_dVL_V);
  DECL_VLA_PTR_PT(T, AP, [SS1][S2 * S2], t_AP);
  DECL_VLA_PTR_PT(short, APD_mask, [SS1][(S2 * S2 + 15) / 16], t_APD_mask);
  DECL_VLA_PTR_PT(T, dCL, [N][S2 * H], t_dCL);
  DECL_VLA_PTR_PT(T, dCL_V, [S2 * H], t_dCL_V);
  DECL_VLA_PTR_PT(T, APD_T, [SS1][S2 * S2], t_APD_T);
  DECL_VLA_PTR_PT(T, dAPO, [SS1][S2 * S2], t_dAPO);
  DECL_VLA_PTR_PT(T, dAPD_V, [SS1][S2 * S2], t_dAPD_V);
  DECL_VLA_PTR_PT(T, HS_T, [H * S2], t_HS_T);
  DECL_VLA_PTR_PT(T, EHS_T, [H * S2], t_EHS_T);
  DECL_VLA_PTR_PT(T, dHS, [N][S2 * H], t_dHS);
  DECL_VLA_PTR_PT(T, dEHS, [N][S2 * H], t_dEHS);
  auto offs = t_offs.data_ptr<int64_t>();
  auto offs2 = t_offs2.data_ptr<int64_t>();

  auto cw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      H,
      S2,
      S2 * S2,
      dt_bf16 ? S2 * H : N * S2 * H,
      0.0,
      XformTPP::XFORM_NONE_TPP,
      0 /*a_trans_flag*/, // We transpose in FWD to have fixed stride of blocks
      1)));
  auto cw_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto a_convert_tpp = SCOPEIT((ConvertTPP<T, float>(S2, S2)), EW_COPY);
  auto ci_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
      S2,
      S2,
      H,
      S2 * H,
      S2 * H,
      dAPO ? 1.0 : 0.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      1)));
  auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<float>(S2 * S2, p), DROPOUT);
  auto softmax_bwd_tpp =
      SCOPEIT((VarSoftMaxBwdTPP<float, float, T>(S2, S2)), SOFTMAX);
  auto scale_tpp = SCOPEIT((ScaleTPP<float, T>(S2 * S2)), EW_SCL);
  auto a_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto ai_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));
  auto aw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      S2,
      S2,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      S2 * S2,
      0.0,
      XformTPP::XFORM_XPOSE_TPP,
      a_trans_flag,
      1)));
  auto vi_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto ki_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto qi_gemm_tpp = (null_EHS ? ki_gemm_tpp : vi_gemm_tpp);
  auto dw_set_zero_tpp = SCOPEIT(SetZeroTPP<T>(H * H), EW_ZERO);
  auto dw_cpy_tpp = SCOPEIT(CpyTPP<T>(H * H), VNNI);
  auto dw_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(H, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto qkvw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      H,
      S2,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      1.0,
      XformTPP::XFORM_NONE_TPP, //(XformTPP::XFORM_TYPE)grad_wt_flag,
      a_trans_flag,
      BS)));
  auto set_zero_dw_tpp = SCOPEIT(SetZeroTPP<T>(H * H), EW_ZERO);
  auto set_zero_f32_tpp = SCOPEIT(SetZeroTPP<float>(N * H), EW_ZERO);
  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, H), BIAS);

  // printf("dAPO = %p, t_dAPO.size = %lu\n", dAPO, t_dAPO.numel());
  //#define PRINT_T(x) std::cout << #x << ": " << x << std::endl
  //#define PRINT_T(x)
#if 0
  {
    RECORD_SCOPE(dwc_gemm, {t_APD_T, t_dCL_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dVL = APD_T * dCL
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          int64_t start = offs[b];
          int64_t ss1 = offs2[b];
          int64_t end = offs[b + 1];
          int64_t len = end - start;
          for (int s21 = start; s21 < end; s21++, ss1 += len) {
            cw_gemm_tpp(APD_T[n][ss1], dCL_V[atrans_blk(start,n)], dVL[s21][n], len);
            if (dt_bf16)
              cw_n2v_tpp(dVL[s21][n], dVL_V[atrans_blk(s21,n)]);
          }
        }
      }
    }
  }
  // PRINT_T(t_AP);
  {
    RECORD_SCOPE(dica_gemm, {t_dCL, t_VL_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dAPD = dCL * VL_TV
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          int64_t start = offs[b];
          int64_t ss1 = offs2[b];
          int64_t end = offs[b + 1];
          int64_t len = end - start;
          for (int s11 = start; s11 < end; s11++, ss1 += len) {
            float dtAPD[len][S2][S2] = {0};
            T dtAPD_bf[len][S2][S2] = {0};
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              if (dAPO)
                a_convert_tpp(dAPO[n][ss1 + ls21], dtAPD[ls21][0]);
              ci_gemm_tpp(dCL[s11][n], VL_TV[s21][n], dtAPD[ls21][0], 1);
            }
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              PCL_ASSERT(0, "t_HM used");
              // t_dAPD[b][s11][n] = t_dAPD[b][s11][n] * t_HM[b][s11][n];
            }
            if (p > 0) {
              for (int l = 0; l < len; l++) {
                dropout_bwd_tpp(dtAPD[l][0], dtAPD[l][0], APD_mask[n][ss1 + l]);
              }
            }
            softmax_bwd_tpp(len, dtAPD[0][0], dtAPD[0][0], AP[n][ss1]);
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              int64_t l = s11 - start;
              int64_t ss = offs2[b];
              scale_tpp(dtAPD[ls21][0], dtAPD_bf[ls21][0], one_by_sqrt_H);
              a_n2v_tpp(dtAPD_bf[ls21][0], dAPD_V[n][ss + ls21 * len + l]);
            }
            // dQL = dADP * KL_V
            ai_gemm_tpp(dtAPD_bf[0][0], KL_V[start][n], dQL[s11][n], len);
            if (dt_bf16)
              cw_n2v_tpp(dQL[s11][n], dQL_V[atrans_blk(s11,n)]);
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(dwa_gemm, {t_QL_T, t_dAPD_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          int64_t start = offs[b];
          int64_t ss1 = offs2[b];
          int64_t end = offs[b + 1];
          int64_t len = end - start;
          for (int s21 = start; s21 < end; s21++, ss1 += len) {
            // dKL = (QL_T * dAPD)T
            aw_gemm_tpp(QL_T[atrans_blk(start,n)], dAPD_V[n][ss1], dKL[s21][n], len);
            if (dt_bf16)
              cw_n2v_tpp(dKL[s21][n], dKL_V[atrans_blk(s21,n)]);
          }
        }
      }
    }
  }
#else
  {
    RECORD_SCOPE(dac_gemm, {t_APD_T, t_dCL_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dVL = APD_T * dCL
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          int64_t start = offs[b];
          // int64_t ss1 = offs2[b];
          int64_t end = offs[b + 1];
          int64_t len = end - start;
          cw_gemm_tpp.config();
          for (int s21 = start, ss1 = offs2[b]; s21 < end; s21++, ss1 += len) {
            cw_gemm_tpp(
                APD_T[n][ss1],
                dCL_V[atrans_blk(start, n)],
                dVL[s21][n],
                len,
                true);
            if (dt_bf16)
              cw_n2v_tpp(dVL[s21][n], dVL_V[atrans_blk(s21, n)]);
          }
          if (!S2_eq_H)
            cw_gemm_tpp.release();
          for (int s11 = start, ss1 = offs2[b]; s11 < end; s11++, ss1 += len) {
            float dtAPD[len][S2][S2];
            T dtAPD_bf[len][S2][S2];
            if (!S2_eq_H)
              ci_gemm_tpp.config();
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              if (dAPO)
                a_convert_tpp(dAPO[n][ss1 + ls21], dtAPD[ls21][0]);
              ci_gemm_tpp(dCL[s11][n], VL_TV[s21][n], dtAPD[ls21][0], 1, true);
            }
            if (!S2_eq_H)
              ci_gemm_tpp.release();
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              PCL_ASSERT(0, "t_HM used");
              // t_dAPD[b][s11][n] = t_dAPD[b][s11][n] * t_HM[b][s11][n];
            }
            if (p > 0) {
              for (int l = 0; l < len; l++) {
                dropout_bwd_tpp(dtAPD[l][0], dtAPD[l][0], APD_mask[n][ss1 + l]);
              }
            }
            softmax_bwd_tpp(len, dtAPD[0][0], dtAPD[0][0], AP[n][ss1]);
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              int64_t l = s11 - start;
              int64_t ss = offs2[b];
              scale_tpp(dtAPD[ls21][0], dtAPD_bf[ls21][0], one_by_sqrt_H);
              a_n2v_tpp(dtAPD_bf[ls21][0], dAPD_V[n][ss + ls21 * len + l]);
            }
            // dQL = dADP * KL_V
            ai_gemm_tpp(
                dtAPD_bf[0][0], KL_V[start][n], dQL[s11][n], len, S2_eq_H);
            if (dt_bf16)
              cw_n2v_tpp(dQL[s11][n], dQL_V[atrans_blk(s11, n)]);
          }
          if (!S2_eq_H)
            aw_gemm_tpp.config();
          for (int s21 = start, ss1 = offs2[b]; s21 < end; s21++, ss1 += len) {
            // dKL = (QL_T * dAPD)T
            aw_gemm_tpp(
                QL_T[atrans_blk(start, n)],
                dAPD_V[n][ss1],
                dKL[s21][n],
                len,
                true);
            if (dt_bf16)
              cw_n2v_tpp(dKL[s21][n], dKL_V[atrans_blk(s21, n)]);
          }
          // The if condition below is just to match config / release on same
          // tpp
          if (!S2_eq_H) {
            aw_gemm_tpp.release();
          } else {
            cw_gemm_tpp.release();
          }
        }
      }
    }
  }
#endif
  // PRINT_T(t_QL_T.permute({0,1,2,4,3}).contiguous());
  // PRINT_T(t_dAPD_V.permute({0,1,2,3,4,6,5}).contiguous().view({B,S1,N,S1,S2,S2}));
  // PRINT_T(t_dKL);
  auto qkv_loop = ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, "bA");
  {
    RECORD_SCOPE(div_gemm, {t_dVL, t_Wv_TV});
    {
#if 0
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          vi_gemm_tpp(dVL[s1][0], Wv_TV[nc][0], dEHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            DECL_VLA_PTR_PT(T, dVL, [N][S2 * H], t_dVL);
            DECL_VLA_PTR_PT(T, Wv_TV, [N][H * H], t_Wv_TV);
            DECL_VLA_PTR_PT(T, dEHS, [N][S2 * H], t_dEHS);
            vi_gemm_tpp(dVL[s1][0], Wv_TV[nc][0], dEHS[s1][nc], N, true);
          },
          [&]() { vi_gemm_tpp.config(); },
          [&]() { vi_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(dik_gemm, {t_dKL, t_Wk_TV});
    {
#if 0
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          ki_gemm_tpp(dKL[s1][0], Wk_TV[nc][0], dEHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            DECL_VLA_PTR_PT(T, dKL, [N][S2 * H], t_dKL);
            DECL_VLA_PTR_PT(T, Wk_TV, [N][H * H], t_Wk_TV);
            DECL_VLA_PTR_PT(T, dEHS, [N][S2 * H], t_dEHS);
            ki_gemm_tpp(dKL[s1][0], Wk_TV[nc][0], dEHS[s1][nc], N, true);
          },
          [&]() { ki_gemm_tpp.config(); },
          [&]() { ki_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(diq_gemm, {t_dQL, t_Wq_TV});
    {
#if 0
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          qi_gemm_tpp(dQL[s1][0], Wq_TV[nc][0], dHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            DECL_VLA_PTR_PT(T, dQL, [N][S2 * H], t_dQL);
            DECL_VLA_PTR_PT(T, Wq_TV, [N][H * H], t_Wq_TV);
            DECL_VLA_PTR_PT(T, dHS, [N][S2 * H], t_dHS);
            qi_gemm_tpp(dQL[s1][0], Wq_TV[nc][0], dHS[s1][nc], N, true);
          },
          [&]() { qi_gemm_tpp.config(); },
          [&]() { qi_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(dwqkv_gemm, {t_HS_T, t_dQL_V});
#if 0
    for (int s1 = 0; s1 < S1; s1 += BS) {
      int count = (s1 + BS <= S1 ? BS : S1 - s1);
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int nk = 0; nk < N; nk++) {
        for (int nc = 0; nc < N; nc++) {
          if (s1 == 0) {
            set_zero_dw_tpp(dWv[nk][nc]);
            set_zero_dw_tpp(dWk[nk][nc]);
            set_zero_dw_tpp(dWq[nk][nc]);
          }
          qkvw_gemm_tpp(EHS_T[atrans_blk(s1,nc)], dVL_V[atrans_blk(s1,nk)], dWv[nk][nc], count);
          qkvw_gemm_tpp(EHS_T[atrans_blk(s1,nc)], dKL_V[atrans_blk(s1,nk)], dWk[nk][nc], count);
          qkvw_gemm_tpp(HS_T[atrans_blk(s1,nc)],  dQL_V[atrans_blk(s1,nk)], dWq[nk][nc], count);
        }
      }
    }
#else
    auto qkvw_loop = ThreadedLoop<3>(
        {LoopSpecs{0, S1, BS, true}, LoopSpecs{N}, LoopSpecs{N}}, "aBC");
    qkvw_loop(
        [&](int* ind) {
          int s1 = ind[0], nk = ind[1], nc = ind[2];
          int count = (s1 + BS <= S1 ? BS : S1 - s1);
          bool is_last_iter = !(s1 + BS < S1);
          DECL_VLA_PTR_PT(T, dWv, [N][H * H], t_dWv);
          DECL_VLA_PTR_PT(T, dWk, [N][H * H], t_dWk);
          DECL_VLA_PTR_PT(T, dWq, [N][H * H], t_dWq);
          DECL_VLA_PTR_PT(T, EHS_T, [H * S2], t_EHS_T);
          DECL_VLA_PTR_PT(T, HS_T, [H * S2], t_HS_T);
          DECL_VLA_PTR_PT(T, dVL_V, [S2 * H], t_dVL_V);
          DECL_VLA_PTR_PT(T, dKL_V, [S2 * H], t_dKL_V);
          DECL_VLA_PTR_PT(T, dQL_V, [S2 * H], t_dQL_V);
          if (s1 == 0) {
            set_zero_dw_tpp(dWv[nk][nc]);
            set_zero_dw_tpp(dWk[nk][nc]);
            set_zero_dw_tpp(dWq[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dVL_V[atrans_blk(s1, nk)],
              dWv[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWv[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWv[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dKL_V[atrans_blk(s1, nk)],
              dWk[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWk[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWk[nk][nc]);
          }
          qkvw_gemm_tpp(
              HS_T[atrans_blk(s1, nc)],
              dQL_V[atrans_blk(s1, nk)],
              dWq[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWq[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWq[nk][nc]);
          }
        },
        [&]() { qkvw_gemm_tpp.config(); },
        [&]() { qkvw_gemm_tpp.release(); });
#endif
  }
  // PRINT_T(t_EHS_T.permute({0,1,2,4,3}).contiguous());
  // PRINT_T(t_HS_T.permute({0,1,2,4,3}).contiguous());
  {
    RECORD_SCOPE(dqkv_bias, {t_dQL});
    int num_threads = omp_get_max_threads();
    float* bias_ptrs[num_threads];
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[N][H];
        bias_ptrs[tid] = prv_grad_bias[0];
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dQL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBq[0]);
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dKL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBk[0]);
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dVL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBv[0]);
      }
    }
  }
  if (null_EHS) {
    t_dEHS = at::Tensor();
  }
}
return std::vector<at::Tensor>(
    {t_dWq, t_dBq, t_dWk, t_dBk, t_dWv, t_dBv, t_dHS, t_dEHS});
