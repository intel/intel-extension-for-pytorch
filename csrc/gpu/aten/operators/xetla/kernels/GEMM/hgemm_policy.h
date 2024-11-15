#pragma once

// clang-format on

#define HGEMM_COMMA ,

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  "_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K "_" #SLM_KS              \
  "_" #B_ROW_MAJOR "_"
#define HGEMM_POLICY_NAME_SYMBOL(                      \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  _##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

struct GemmShapeT {
  int m_, n_, k_;
  size_t operator()(const GemmShapeT& t) const {
    size_t seed = 0;
    seed ^= std::hash<int>()(t.m_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(t.n_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(t.k_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
  size_t operator==(const GemmShapeT& other) const {
    return m_ == other.m_ && n_ == other.n_ && k_ == other.k_;
  }
};

static inline int hgemm_find_policy_id_(
    const int m,
    const int n,
    const int k,
    std::unordered_map<GemmShapeT, int, GemmShapeT>& special_table) {
  auto it = special_table.find(GemmShapeT{m, n, k});
  if (it != special_table.end()) {
    int idx = it->second;
    return idx;
  }

  return -1;
}

#if 0
struct GemmPolicyT {
  int wg_m_, wg_n_, sg_m_, sg_n_, sg_k_, slm_ks_, l3_ks_;
  bool is_b_row_major_;
  GemmPolicyT(
      int wg_m,
      int wg_n,
      int sg_m,
      int sg_n,
      int sg_k,
      int slm_ks,
      bool is_b_row_major)
      : wg_m_(wg_m),
        wg_n_(wg_n),
        sg_m_(sg_m),
        sg_n_(sg_n),
        sg_k_(sg_k),
        slm_ks_(slm_ks),
        is_b_row_major_(is_b_row_major) {
    l3_ks_ = 1;
  }
};

#define HGEMM_POLICY_TRAITS(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  { WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR }
#endif
