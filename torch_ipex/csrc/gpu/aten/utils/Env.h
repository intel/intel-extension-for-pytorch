#pragma once
#include <c10/util/Exception.h>

enum DPCPP_ENV {
  ENV_VERBOSE = 0,
  ENV_FORCE_SYNC,
  ENV_DISABLE_PROFILING,
  ENV_LAZY_REORDER,
  ENV_WEIGHT_CACHE,
  ENV_TILE_AS_DEVICE };

int dpcpp_env(int env);

static inline int dpcpp_verbose() {
  return dpcpp_env(ENV_VERBOSE);
}

static inline int dpcpp_force_sync() {
  return dpcpp_env(ENV_FORCE_SYNC);
}

static inline bool dpcpp_profiling() {
  return !dpcpp_env(ENV_DISABLE_PROFILING);
}

static inline int lazy_reorder_enabled() {
  return dpcpp_env(ENV_LAZY_REORDER);
}

static inline int weight_cache_enabled() {
  auto weight_cache_env = dpcpp_env(ENV_WEIGHT_CACHE);
  if (weight_cache_env) {
    TORCH_CHECK(
        lazy_reorder_enabled() == 1,
        "IPEX_WEIGHT_CACHE can be set only when IPEX_LAZY_REORDER=1.");
  }
  return weight_cache_env;
}

static inline bool tile_as_device() {
  return (bool)dpcpp_env(ENV_TILE_AS_DEVICE);
}
