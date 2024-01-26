RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
int i = 0;
auto t_in_ids = inputs[i++]; // [B][S]
auto t_pos_ids = inputs[i++]; // [1][S]
auto t_tt_ids = inputs[i++]; // [B][S]
auto t_in_emb = inputs[i++]; // [B][S][NH]
auto t_gamma = inputs[i++]; // [NH]
auto t_beta = inputs[i++]; // [NH]
auto t_word_emb = inputs[i++]; // [*][NH]
auto t_pos_emb = inputs[i++]; // [*][NH]
auto t_tt_emb = inputs[i++]; // [*][NH]

int64_t B, S1, N, S2;
bool in_ids_null = t_in_ids.numel() == 0;
bool tt_ids_null = t_tt_ids.numel() == 0;
bool pos_ids_null = t_pos_ids.numel() == 0;
bool in_emb_null = t_in_emb.numel() == 0;

PCL_ASSERT(
    (!in_ids_null || !in_emb_null),
    "Either Input_ids or input_embeddings must be non-empty");

if (in_emb_null == false) {
  auto in_sizes = t_in_emb.sizes();
  B = in_sizes[0];
  S1 = in_sizes[1];
  N = in_sizes[2];
  S2 = in_sizes[3];
  // H = in_sizes[4];
} else {
  auto in_sizes = t_in_ids.sizes();
  B = in_sizes[0];
  S1 = in_sizes[1];
  S2 = in_sizes[2];
  int64_t NH = t_gamma.size(0);
  N = NH / H;
}

auto t_out = t_gamma.new_empty({B, S1, N, S2, H});
auto t_emb_out = t_out;
if (training)
  t_emb_out = t_gamma.new_empty({B, S1, N, S2, H});
auto t_mean = t_gamma.new_empty({B, S1, S2}, at::kFloat);
auto t_var = t_gamma.new_empty({B, S1, S2}, at::kFloat);

// auto t_dp_mask = at::empty({B, S1, (N*S2*H+15)/16}, at::kShort);
auto t_dp_mask = at::empty({0}, at::kShort);

if (p > 0)
  t_dp_mask = at::empty({B, S1, (N * S2 * H + 15) / 16}, at::kShort);

DECL_VLA_PTR_PT(int64_t, in_ids, [S1][S2], t_in_ids);
DECL_VLA_PTR_PT(int64_t, pos_ids, [S1][S2], t_pos_ids);
DECL_VLA_PTR_PT(int64_t, tt_ids, [S1][S2], t_tt_ids);
DECL_VLA_PTR_PT(T, in_emb, [S1][N][S2][H], t_in_emb);
DECL_VLA_PTR_PT(T, gamma, [H], t_gamma);
DECL_VLA_PTR_PT(T, beta, [H], t_beta);
DECL_VLA_PTR_PT(float, mean, [S1][S2], t_mean);
DECL_VLA_PTR_PT(float, var, [S1][S2], t_var);
DECL_VLA_PTR_PT(T, emb_out, [S1][N][S2][H], t_emb_out);
DECL_VLA_PTR_PT(T, out, [S1][N][S2][H], t_out);
DECL_VLA_PTR_PT(short, dp_mask, [S1][(N * S2 * H + 15) / 16], t_dp_mask);
DECL_VLA_PTR_PT(ET, word_emb, [N][H], t_word_emb);
DECL_VLA_PTR_PT(ET, pos_emb, [N][H], t_pos_emb);
DECL_VLA_PTR_PT(ET, tt_emb, [N][H], t_tt_emb);

auto layer_norm_fwd_tpp =
    SCOPEIT(LayerNormFwdTPP<T>(N, S2, H, eps), LAYER_NORM);
auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(N * S2 * H, p), DROPOUT);

{
  RECORD_SCOPE(b_emb, {t_out, t_word_emb});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int s1 = 0; s1 < S1; s1++) {
        for (int s2 = 0; s2 < S2; s2++) {
          int64_t w_id = -1, pos_id = s1 * S2 + s2, tt_id = 0;
          if (!in_ids_null)
            w_id = in_ids[b][s1][s2];
          if (!pos_ids_null)
            pos_id = pos_ids[b][s1][s2];
          if (!tt_ids_null)
            tt_id = tt_ids[b][s1][s2];
          for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
              float sum = 0.0f;
              if (!in_ids_null) {
                if (w_id != pad_id)
                  sum += word_emb[w_id][n][h];
              } else {
                sum += in_emb[b][s1][n][s2][h];
              }
              sum += pos_emb[pos_id][n][h];
              sum += tt_emb[tt_id][n][h];
              emb_out[b][s1][n][s2][h] = sum;
            }
          }
        }
        layer_norm_fwd_tpp(
            emb_out[b][s1][0][0],
            gamma[0],
            beta[0],
            mean[b][s1],
            var[b][s1],
            out[b][s1][0][0]);
        if (p > 0) {
          dropout_fwd_tpp(
              out[b][s1][0][0],
              (void*)get_rng_state(),
              out[b][s1][0][0],
              dp_mask[b][s1]);
        }
      }
    }
  }
}
return std::vector<at::Tensor>({t_out, t_emb_out, t_mean, t_var, t_dp_mask});
