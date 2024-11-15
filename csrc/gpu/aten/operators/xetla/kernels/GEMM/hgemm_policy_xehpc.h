#pragma once

#ifdef USE_XETLA_XE_HPC

#include "hgemm_policy.h"

// clang-format off
#define HGEMM_NUM_B_ROW_POLICIES_ 30
#define HGEMM_ENUMERATE_POLICIES_(_, B_ROW_MAJOR, T) \
  _(8, 64, 8, 16, 32, 8, B_ROW_MAJOR)T      \
  _(8, 64, 8, 16, 16, 4, B_ROW_MAJOR)T      \
  _(8, 32, 8, 16, 16, 4, B_ROW_MAJOR)T      \
  _(8, 32, 8, 16, 16, 8, B_ROW_MAJOR)T      \
  _(8, 128, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 128, 8, 16, 32, 4, B_ROW_MAJOR)T     \
  _(8, 256, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 512, 8, 16, 16, 1, B_ROW_MAJOR)T     \
  _(16, 64, 16, 16, 16, 8, B_ROW_MAJOR)T    \
  _(16, 256, 8, 16, 16, 1, B_ROW_MAJOR)T    \
  _(16, 256, 16, 16, 16, 2, B_ROW_MAJOR)T   \
  _(16, 512, 16, 16, 16, 1, B_ROW_MAJOR)T   \
  _(32, 128, 8, 16, 32, 1, B_ROW_MAJOR)T    \
  _(32, 64, 32, 16, 16, 8, B_ROW_MAJOR)T    \
  _(32, 64, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(32, 128, 32, 16, 16, 4, B_ROW_MAJOR)T   \
  _(32, 256, 32, 16, 16, 2, B_ROW_MAJOR)T   \
  _(32, 512, 32, 16, 16, 1, B_ROW_MAJOR)T   \
  _(64, 128, 64, 16, 16, 4, B_ROW_MAJOR)T   \
  _(64, 256, 64, 16, 16, 2, B_ROW_MAJOR)T   \
  _(64, 512, 64, 16, 16, 1, B_ROW_MAJOR)T   \
  _(128, 128, 32, 32, 32, 2, B_ROW_MAJOR)T  \
  _(128, 256, 64, 16, 16, 1, B_ROW_MAJOR)T  \
  _(128, 512, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 32, 1, B_ROW_MAJOR)T  \
  _(128, 64, 16, 16, 64, 1, B_ROW_MAJOR)T   \
  _(128, 128, 16, 32, 64, 1, B_ROW_MAJOR)T  \
  _(128, 256, 32, 32, 16, 1, B_ROW_MAJOR)T

#define HGEMM_ENUMERATE_POLICIES(_)    \
  HGEMM_ENUMERATE_POLICIES_(_, true, )

#define HGEMM_ENUMERATE_POLICIES_COMMA(_)         \
  HGEMM_ENUMERATE_POLICIES_(_, true, HGEMM_COMMA)

#define HGEMM_NUM_POLICIES_ (HGEMM_NUM_B_ROW_POLICIES_)

// clang-format on

enum hgemm_policy {
  NONE = -1,
  HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_POLICY_NAME_SYMBOL)
};

#define HGEMM_NUM_POLICIES (HGEMM_NUM_POLICIES_)
#endif
