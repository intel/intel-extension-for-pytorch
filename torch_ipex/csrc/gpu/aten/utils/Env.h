#pragma once

enum DPCPP_ENV {
  ENV_VERBOSE = 0,
  ENV_FORCE_SYNC,
  ENV_DISABLE_PROFILING,
  ENV_LAZY_REORDER,
  ENV_WEIGHT_CACHE,
  ENV_TILE_AS_DEVICE,
  ENV_DEV_INDEX };

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
  return dpcpp_env(ENV_WEIGHT_CACHE);
}

static inline bool tile_as_device() {
  return (bool)dpcpp_env(ENV_TILE_AS_DEVICE);
}

static inline int ipex_dev_index() {
  return dpcpp_env(ENV_DEV_INDEX);
}
