RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
// B - Batch size
// S - Max seq len
// N - Number of attention heads
// H - Head size
auto t_Wq = inputs[0]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bq = inputs[1]; // [HS]
auto t_Wk = inputs[2]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bk = inputs[3]; // [HS]
auto t_Wv = inputs[4]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bv = inputs[5]; // [HS]
auto t_HS = inputs[6]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_AM = inputs[7]; // Optional [B][S]
auto t_HM = inputs[8]; // Optional [B][N][S][S]
auto t_EHS = inputs[9]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_EAM = inputs[10]; // Optional [B][S]
auto t_offs = inputs[11]; // [B+1]
auto t_offs2 = inputs[12]; // [B+1]

int64_t B = t_offs.sizes()[0] - 1;
int64_t SS1 = t_offs2[B].item().to<int64_t>();
auto sizes = t_HS.sizes();
int64_t S1 = sizes[0];
int64_t N = sizes[1];
int64_t S2 = sizes[2];
int64_t H = sizes[3];
// int64_t NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
bool null_EHS = false;
bool dt_bf16 = (t_HS.dtype() == at::kBFloat16);
bool bf16_training = (training && dt_bf16);
auto t_EHS_orig = t_EHS;

// std::cout << "B: " << B << " S1: " << S1 << " S2: " << S2 << " N: " << N << "
// H: " << H << std::endl;
if (t_EHS.numel() == 0) {
  null_EHS = true;
  t_EHS = t_HS;
} else {
  t_AM = t_EAM;
}

//#define PRINT_T(x) std::cout << #x << ": " << x << std::endl
auto t_HS_T = t_HS;
auto t_EHS_T = t_EHS;

auto t_Wq_V = wt_tensor_for_fwd(N, H, N, H, t_Wq);
auto t_Wk_V = wt_tensor_for_fwd(N, H, N, H, t_Wk);
auto t_Wv_V = wt_tensor_for_fwd(N, H, N, H, t_Wv);

auto t_QL = t_HS.new_empty({S1, N, S2, H});
auto t_QL_T = t_QL;
auto t_KL_TV = t_EHS.new_empty({S1, N, H, S2});
if (dt_bf16)
  t_KL_TV = t_KL_TV.view({S1, N, H / 2, S2, 2});
auto t_KL_V = t_KL_TV;
auto t_VL_V = t_EHS.new_empty({S1, N, S2, H});
if (dt_bf16)
  t_VL_V = t_VL_V.view({S1, N, S2 / 2, H, 2});
auto t_VL_TV = t_VL_V;
auto t_AP = t_QL.new_empty({N, SS1, S2, S2});
auto t_CL = t_AP.new_empty({S1, N, S2, H});

auto t_APD = t_AP;
auto t_APD_mask = at::empty({N, SS1, (S2 * S2 + 15) / 16}, at::kShort);
if (p > 0 || t_HM.numel() != 0) {
  t_APD = at::empty_like(t_AP);
}

auto t_APD_T = t_APD;

if (bf16_training) {
  t_HS_T = t_HS.new_empty({N, S1, H, S2}); // For BWD only
  t_EHS_T = null_EHS ? t_HS_T : t_HS.new_empty({N, S1, H, S2}); // For BWD only

  t_QL_T = t_HS.new_empty({N, S1, H, S2}); // For BWD only
}
if (training) {
  if (dt_bf16) {
    t_KL_V = t_EHS.new_empty({S1, N, S2 / 2, H, 2}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({S1, N, H / 2, S2, 2}); // For BWD only
  } else {
    t_KL_V = t_EHS.new_empty({S1, N, S2, H}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({S1, N, H, S2}); // For BWD only
  }
  t_APD_T = t_QL.new_empty({N, SS1, S2, S2}); // For BWD only
}

{
  // float (*QL)[S1][N][S2][H] = (float
  // (*)[S1][N][S2][H])t_QL.data_ptr<float>();
  // DECL_VLA_PTR_PT(T, Wq_V, [N][H * H], t_Wq_V);
  //  DECL_VLA_PTR_PT(T, Wk_V, [N][H * H], t_Wk_V);
  DECL_VLA_PTR_PT(T, Wv_V, [N][H * H], t_Wv_V);
  // DECL_VLA_PTR_PT(T, Bq, [H], t_Bq);
  // DECL_VLA_PTR_PT(T, Bk, [H], t_Bk);
  DECL_VLA_PTR_PT(T, Bv, [H], t_Bv);
  DECL_VLA_PTR_PT(T, QL, [N][S2 * H], t_QL);
  // DECL_VLA_PTR_PT(T, QL_T, [S1][H * S2], t_QL_T); // For BWD only
  // DECL_VLA_PTR_PT(T, KL_V, [N][S2 * H], t_KL_V);
  DECL_VLA_PTR_PT(T, KL_TV, [N][H * S2], t_KL_TV);
  DECL_VLA_PTR_PT(T, VL_V, [N][S2 * H], t_VL_V);
  DECL_VLA_PTR_PT(T, VL_TV, [N][H * S2], t_VL_TV);
  DECL_VLA_PTR_PT(T, AP, [SS1][S2 * S2], t_AP);
  DECL_VLA_PTR_PT(T, APD, [SS1][S2 * S2], t_APD);
  DECL_VLA_PTR_PT(T, APD_T, [SS1][S2 * S2], t_APD_T); // For BWD only
  DECL_VLA_PTR_PT(short, APD_mask, [SS1][(S2 * S2 + 15) / 16], t_APD_mask);
  DECL_VLA_PTR_PT(T, CL, [N][S2 * H], t_CL);
  // DECL_VLA_PTR_PT(T, HS, [N][S2 * H], t_HS);
  // DECL_VLA_PTR_PT(T, HS_T, [N][H * S2], t_HS_T); // for BWD only
  DECL_VLA_PTR_PT(T, EHS, [N][S2 * H], t_EHS);
  // DECL_VLA_PTR_PT(T, EHS_T, [N][H * S2], t_EHS_T); // for BWD only
  DECL_VLA_PTR_PT(T, AM, [S2], t_AM);
  auto offs = t_offs.data_ptr<int64_t>();
  auto offs2 = t_offs2.data_ptr<int64_t>();

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, H), BIAS);
  auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto k_xpose_tpp_1 = SCOPEIT(
      XformExtTPP<T>(
          S2,
          H,
          training ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_XPOSE_N2V_TPP,
          true),
      XPOSE);
  auto kv_xpose_tpp_2 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_N2V_TPP, true), VNNI);
  auto v_xpose_tpp_1 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
      S2, S2, H, S2 * H, H * S2, 0.0, XformTPP::XFORM_NONE_TPP, 0, 1)));
  auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(S2 * S2)), EW_SCL);
  auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(S2, S2), EW_ADD);
  auto softmax_fwd_tpp = SCOPEIT((VarSoftMaxFwdTPP<float, T>(S2, S2)), SOFTMAX);
  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S2 * S2, p), DROPOUT);
  auto a_xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));

  {
    RECORD_SCOPE(q_gemm, {t_HS, t_Wq_V});
    {
#if 0
        DECL_VLA_PTR_PT(T, HS, [N][S2 * H], t_HS);
        DECL_VLA_PTR_PT(T, HS_T, [S1][H * S2], t_HS_T); // for BWD only
        DECL_VLA_PTR_PT(T, Bq, [H], t_Bq);
        DECL_VLA_PTR_PT(T, Wq_V, [N][H * H], t_Wq_V);
        DECL_VLA_PTR_PT(T, QL, [N][S2 * H], t_QL);
        DECL_VLA_PTR_PT(T, QL_T, [S1][H * S2], t_QL_T); // For BWD only
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < N; nk++) {
          if (bf16_training && nk == 0)
            xpose_tpp(N, S2 * H, S1 * S2 * H, HS[s1][0], HS_T[0][s1]);
          copy_bias_tpp(Bq[nk], QL[s1][nk]);
          qkv_gemm_tpp(HS[s1][0], Wq_V[nk][0], QL[s1][nk], N);
          if (bf16_training)
            xpose_tpp(QL[s1][nk], QL_T[nk][s1]);
        }
      }
#else
      int64_t BN = N;
      auto qkv_loop = ThreadedLoop<3>(
          {LoopSpecs{0L, N, BN, true}, LoopSpecs{S1}, LoopSpecs{N}}, "acB");
      // ThreadedLoop<3>({LoopSpecs{0L,N,BN}, LoopSpecs{S1}, LoopSpecs{N}},
      // "acB");  ThreadedLoop<3>({LoopSpecs{0L,N,BN}, LoopSpecs{S1},
      // LoopSpecs{N}}, "aBC");  ThreadedLoop<3>({LoopSpecs{0L,N,BN},
      // LoopSpecs{S1}, LoopSpecs{N, {4}}}, "acBC");
      qkv_loop(
          [&](int* ind) {
            int bn = ind[0], s1 = ind[1], nk = ind[2];
            DECL_VLA_PTR_PT(T, HS, [N][S2 * H], t_HS);
            DECL_VLA_PTR_PT(T, HS_T, [S1][H * S2], t_HS_T); // for BWD only
            DECL_VLA_PTR_PT(T, Bq, [H], t_Bq);
            DECL_VLA_PTR_PT(T, Wq_V, [N][H * H], t_Wq_V);
            DECL_VLA_PTR_PT(T, QL, [N][S2 * H], t_QL);
            DECL_VLA_PTR_PT(T, QL_T, [S1][H * S2], t_QL_T); // For BWD only
            if (bf16_training && nk == 0)
              xpose_tpp(BN, S2 * H, S1 * S2 * H, HS[s1][bn], HS_T[bn][s1]);
            if (bn == 0)
              copy_bias_tpp(Bq[nk], QL[s1][nk]);
            qkv_gemm_tpp(HS[s1][bn], Wq_V[nk][bn], QL[s1][nk], BN, true);
            if (bf16_training)
              if (bn == N - BN)
                xpose_tpp(QL[s1][nk], QL_T[nk][s1]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }

  // PRINT_T(t_QL.permute({0,1,3,2,4}).contiguous().view({B,S1*S2,N*H}));

  {
    RECORD_SCOPE(k_gemm, {t_EHS, t_Wk_V});
    {
#if 0
        DECL_VLA_PTR_PT(T, EHS_T, [S1][H * S2], t_EHS_T); // for BWD only
        DECL_VLA_PTR_PT(T, KL_V, [N][S2 * H], t_KL_V);
        DECL_VLA_PTR_PT(T, Bk, [H], t_Bk);
        DECL_VLA_PTR_PT(T, Wk_V, [N][H * H], t_Wk_V);
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < N; nk++) {
          T tmp[S2 * H];
          T* tmpp = (training && !bf16_training) ? KL_V[s1][nk] : tmp;
          if (!null_EHS && bf16_training && nk == 0)
            xpose_tpp(N, S2 * H, S1 * S2 * H, EHS[s1][0], EHS_T[0][s1]);
          copy_bias_tpp(Bk[nk], tmpp);
          qkv_gemm_tpp(EHS[s1][0], Wk_V[nk][0], tmpp, N);
          k_xpose_tpp_1(tmpp, KL_V[s1][nk]); // KL_V = KL_VT if not training
          if (training)
            kv_xpose_tpp_2(tmpp, KL_TV[s1][nk]);
        }
      }
#else
      auto qkv_loop = ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, "bA");
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];
            DECL_VLA_PTR_PT(T, Bk, [H], t_Bk);
            DECL_VLA_PTR_PT(T, Wk_V, [N][H * H], t_Wk_V);
            DECL_VLA_PTR_PT(T, KL_V, [N][S2 * H], t_KL_V);
            DECL_VLA_PTR_PT(T, KL_TV, [N][H * S2], t_KL_TV);
            DECL_VLA_PTR_PT(T, EHS, [N][S2 * H], t_EHS);
            DECL_VLA_PTR_PT(T, EHS_T, [S1][H * S2], t_EHS_T); // for BWD only

            T tmp[S2 * H];
            T* tmpp = (training && !bf16_training) ? KL_V[s1][nk] : tmp;
            if (!null_EHS && bf16_training && nk == 0)
              xpose_tpp(N, S2 * H, S1 * S2 * H, EHS[s1][0], EHS_T[0][s1]);
            copy_bias_tpp(Bk[nk], tmpp);
            qkv_gemm_tpp(EHS[s1][0], Wk_V[nk][0], tmpp, N);
            k_xpose_tpp_1(tmpp, KL_V[s1][nk]); // KL_V = KL_VT if not training
            if (training)
              kv_xpose_tpp_2(tmpp, KL_TV[s1][nk]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }
  // PRINT_T(t_EHS);
  // PRINT_T(t_Wk_V.permute({0,1,2,4,3}).contiguous().view({N,N,H,H}));
  // PRINT_T(t_Wk_V);
  // PRINT_T(t_Bk);
  // PRINT_T(t_KL_V.permute({0,1,3,5,2,4}).contiguous().view({B,S1*S2,N*H}));

  {
    RECORD_SCOPE(v_gemm, {t_EHS, t_Wv_V});
    {
#if 0
      DECL_VLA_PTR_PT(T, EHS, [N][S2 * H], t_EHS);
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < N; nk++) {
          T tmp[S2 * H];
          T* tmpp = (!dt_bf16) ? VL_V[s1][nk] : tmp;
          copy_bias_tpp(Bv[nk], tmpp);
          qkv_gemm_tpp(EHS[s1][0], Wv_V[nk][0], tmpp, N);
          v_xpose_tpp_1(tmpp, VL_V[s1][nk]);
          if (training)
            kv_xpose_tpp_2(tmpp, VL_TV[s1][nk]);
        }
      }
#else
      auto qkv_loop = ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, "bA");
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];
            DECL_VLA_PTR_PT(T, Bv, [H], t_Bv);
            DECL_VLA_PTR_PT(T, Wv_V, [N][H * H], t_Wv_V);
            DECL_VLA_PTR_PT(T, VL_V, [N][S2 * H], t_VL_V);
            DECL_VLA_PTR_PT(T, VL_TV, [N][H * S2], t_VL_TV);
            DECL_VLA_PTR_PT(T, EHS, [N][S2 * H], t_EHS);

            T tmp[S2 * H];
            T* tmpp = (!dt_bf16) ? VL_V[s1][nk] : tmp;
            copy_bias_tpp(Bv[nk], tmpp);
            qkv_gemm_tpp(EHS[s1][0], Wv_V[nk][0], tmpp, N);
            v_xpose_tpp_1(tmpp, VL_V[s1][nk]);
            if (training)
              kv_xpose_tpp_2(tmpp, VL_TV[s1][nk]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }
  // Take the dot product between "query" and "key" to get the raw attention
  // scores.
  {
    RECORD_SCOPE(ac_gemm, {t_QL, t_KL_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#ifndef _WIN32 // TODO: Fix crash on ICX Windows. CMPLRLLVM-55384
#pragma omp parallel for collapse(2) schedule(static, 1)
#else
#pragma omp for
#endif
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          int64_t start = offs[b];
          int64_t ss1 = offs2[b];
          int64_t end = offs[b + 1];
          int64_t len = end - start;
          for (int s11 = start; s11 < end; s11++, ss1 += len) {
            float AS[len][S2][S2];
            for (int s21 = start; s21 < end; s21++) {
              int64_t ls21 = s21 - start;
              a_gemm_tpp(QL[s11][n], KL_TV[s21][n], AS[ls21][0], 1);
              scale_tpp(AS[ls21][0], AS[ls21][0], one_by_sqrt_H);
              if (t_AM.numel() != 0)
                add_mask_tpp(AM[s21], AS[ls21][0]);
            }
            softmax_fwd_tpp(len, AS[0][0], AP[n][ss1]);
            if (p > 0) {
              for (int l = 0; l < len; l++) {
                dropout_fwd_tpp(
                    AP[n][ss1 + l],
                    rng_state,
                    APD[n][ss1 + l],
                    APD_mask[n][ss1 + l]);
              }
            }
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              PCL_ASSERT(0, "t_HM used");
              // t_APD[b][s11][n] *= t_HM[b][s11][n];
            }
            if (training) {
              int64_t l = s11 - start;
              int64_t ss = offs2[b];
              // xpose S1xS1 part as well here to allow fix stride in GEMM in
              // bwd
              a_xpose_tpp(
                  len, S2 * S2, len * S2 * S2, APD[n][ss1], APD_T[n][ss + l]);
            }
            c_gemm_tpp(APD[n][ss1], VL_V[start][n], CL[s11][n], len);
          }
        }
      }
    }
  }
}
// auto t_APO = t_APD.permute({0, 2, 1, 4, 3, 5}).contiguous().view({B, N, S,
// S});
auto t_APO = t_APD;
return std::vector<at::Tensor>(
    {t_CL,
     t_APO,
     t_HS_T,
     null_EHS ? t_EHS_orig : t_EHS_T,
     t_QL_T,
     t_KL_V,
     t_VL_TV,
     t_AP,
     t_APD_T,
     t_APD_mask});
