RECORD_FUNCTION("bert_bwd", std::vector<c10::IValue>());
int i = 0;
auto t_grad_out = inputs[i++]; // [B][S1][N][S2][H]
auto t_in_ids = inputs[i++]; // [B][S]
auto t_pos_ids = inputs[i++]; // [1][S]
auto t_tt_ids = inputs[i++]; // [B][S]
auto t_in_emb = inputs[i++]; // [B][S]
auto t_gamma = inputs[i++]; // [NH]
auto t_word_emb = inputs[i++]; // [*][NH]
auto t_pos_emb = inputs[i++]; // [*][NH]
auto t_tt_emb = inputs[i++]; // [*][NH]
auto t_mean = inputs[i++]; // [N][H]
auto t_var = inputs[i++]; // [N][H]
auto t_emb_out = inputs[i++]; // [N][H]
auto t_dp_mask = inputs[i++];

int64_t B, S1, N, S2, H;
// bool in_ids_null = t_in_ids.numel() == 0;
bool tt_ids_null = t_tt_ids.numel() == 0;
bool pos_ids_null = t_pos_ids.numel() == 0;
bool in_emb_null = t_in_emb.numel() == 0;

auto in_sizes = t_grad_out.sizes();
B = in_sizes[0];
S1 = in_sizes[1];
N = in_sizes[2];
S2 = in_sizes[3];
H = in_sizes[4];

auto t_grad_in_emb = at::empty_like(t_in_emb);
auto t_grad_gamma = at::empty_like(t_gamma); // [N][H]
auto t_grad_beta = at::empty_like(t_gamma); // [N][H]
auto t_grad_word_emb = at::empty_like(t_word_emb);
auto t_grad_pos_emb = at::empty_like(t_pos_emb);
auto t_grad_tt_emb = at::empty_like(t_tt_emb);
auto t_grad_dp_out = t_grad_out;
auto t_grad_emb_out = at::empty_like(t_emb_out);
if (p > 0) {
  t_grad_dp_out = t_grad_emb_out;
}

DECL_VLA_PTR_PT(int64_t, in_ids, [S1][S2], t_in_ids);
DECL_VLA_PTR_PT(int64_t, pos_ids, [S1][S2], t_pos_ids);
DECL_VLA_PTR_PT(int64_t, tt_ids, [S1][S2], t_tt_ids);
DECL_VLA_PTR_PT(T, grad_in_emb, [S1][N][S2][H], t_grad_in_emb);
DECL_VLA_PTR_PT(T, gamma, [H], t_gamma);
DECL_VLA_PTR_PT(T, grad_gamma, [H], t_grad_gamma);
DECL_VLA_PTR_PT(T, grad_beta, [H], t_grad_beta);
DECL_VLA_PTR_PT(float, mean, [S1][S2], t_mean);
DECL_VLA_PTR_PT(float, var, [S1][S2], t_var);
DECL_VLA_PTR_PT(T, emb_out, [S1][N][S2][H], t_emb_out);
DECL_VLA_PTR_PT(T, grad_out, [S1][N][S2][H], t_grad_out);
DECL_VLA_PTR_PT(T, grad_dp_out, [S1][N][S2][H], t_grad_dp_out);
DECL_VLA_PTR_PT(T, grad_emb_out, [S1][N][S2][H], t_grad_emb_out);
DECL_VLA_PTR_PT(short, dp_mask, [S1][(N * S2 * H + 15) / 16], t_dp_mask);
DECL_VLA_PTR_PT(ET, grad_word_emb, [N][H], t_grad_word_emb);
DECL_VLA_PTR_PT(ET, grad_pos_emb, [N][H], t_grad_pos_emb);
DECL_VLA_PTR_PT(ET, grad_tt_emb, [N][H], t_grad_tt_emb);

auto drop_out_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(N * S2 * H, p), DROPOUT);
auto layer_norm_bwd_tpp = SCOPEIT(LayerNormBwdTPP<T>(N, S2, H), LAYER_NORM);
auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(N * H), EW_ZERO);

{
  RECORD_SCOPE(db_emb, {t_grad_out, t_word_emb});
#if 0
    t_grad_gamma.zero_();
    t_grad_beta.zero_();
#else
  tensor_set_zero(N, H, t_grad_gamma);
  tensor_set_zero(N, H, t_grad_beta);
#endif
  int num_threads = omp_get_max_threads();
  float* gamma_ptrs[num_threads];
  float* beta_ptrs[num_threads];
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_gamma[N][H];
      float prv_grad_beta[N][H];
      beta_ptrs[tid] = prv_grad_beta[0];
      gamma_ptrs[tid] = prv_grad_gamma[0];
      set_zero_tpp(prv_grad_gamma[0]);
      set_zero_tpp(prv_grad_beta[0]);
#pragma omp for collapse(2)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          // T grad_emb_out[N][S2][H];
          if (p > 0) {
            drop_out_bwd_tpp(
                grad_out[b][s1][0][0],
                grad_dp_out[b][s1][0][0],
                dp_mask[b][s1]);
          }
          layer_norm_bwd_tpp(
              grad_dp_out[b][s1][0][0],
              emb_out[b][s1][0][0],
              mean[b][s1],
              var[b][s1],
              gamma[0],
              grad_emb_out[b][s1][0][0],
              prv_grad_gamma[0],
              prv_grad_beta[0]);
        }
      }
#pragma omp barrier
      omp_reduce_buf(num_threads, N * H, gamma_ptrs, grad_gamma[0]);
      omp_reduce_buf(num_threads, N * H, beta_ptrs, grad_beta[0]);
    }
  }
}
{
  RECORD_SCOPE(db_emb, {t_grad_out, t_word_emb});
  if (in_emb_null)
    t_grad_word_emb.zero_();

  t_grad_pos_emb.zero_();
  t_grad_tt_emb.zero_();
  for (int b = 0; b < B; b++) {
    for (int s1 = 0; s1 < S1; s1++) {
      for (int s2 = 0; s2 < S2; s2++) {
        int64_t w_id = -1, pos_id = s1 * S2 + s2, tt_id = 0;
        if (in_emb_null)
          w_id = in_ids[b][s1][s2];
        if (!pos_ids_null)
          pos_id = pos_ids[b][s1][s2];
        if (!tt_ids_null)
          tt_id = tt_ids[b][s1][s2];
        for (int n = 0; n < N; n++) {
          for (int h = 0; h < H; h++) {
            float grad = grad_emb_out[b][s1][n][s2][h];
            if (in_emb_null) {
              if (w_id != pad_id)
                grad_word_emb[w_id][n][h] += grad;
            } else {
              grad_in_emb[b][s1][n][s2][h] = grad;
            }
            grad_pos_emb[pos_id][n][h] += grad;
            grad_tt_emb[tt_id][n][h] += grad;
          }
        }
      }
    }
  }
}
return std::vector<at::Tensor>(
    {t_grad_in_emb,
     t_grad_gamma,
     t_grad_beta,
     t_grad_word_emb,
     t_grad_pos_emb,
     t_grad_tt_emb});
