#pragma once

enum {ENV_VERBOSE = 0, ENV_FORCE_SYNC, ENV_LAZY_REORDER};

int dpcpp_env(int env);

static inline int lazy_reorder_enabled() {
  return dpcpp_env(ENV_LAZY_REORDER);
}
