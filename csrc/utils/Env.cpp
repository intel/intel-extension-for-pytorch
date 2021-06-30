#include <utils/Env.h>
#include <sstream>
#include <iostream>

enum DPCPP_ENV {
  ENV_VERBOSE = 0,
  ENV_FORCE_SYNC,
  ENV_DISABLE_PROFILING,
  ENV_LAZY_REORDER,
  ENV_DISABLE_TILE_PARTITION };

#define DPCPP_ENV_TYPE_DEF(type, var)                                    \
    int type = [&]() -> int {                                            \
      auto env = std::getenv("IPEX_" #var);                              \
      int _##type = 0;                                                   \
      try {                                                              \
        _##type = std::stoi(env, 0, 10);                                 \
      } catch (...) { /* Do Nothing */ }                                 \
      std::cerr << " ** IPEX_" << #var << ": " << _##type << std::endl;  \
      return _##type;                                                    \
    } ()

int dpcpp_env(int env_type) {
  static auto _header = []() -> bool {
    std::cerr << std::endl
      << "/*********************************************************" << std::endl
      << " ** The values of all available launch options for IPEX **" << std::endl;
    return true;
  } ();

  static struct {
    DPCPP_ENV_TYPE_DEF(verbose, VERBOSE);
    DPCPP_ENV_TYPE_DEF(force_sync, FORCE_SYNC);
    DPCPP_ENV_TYPE_DEF(disable_profiling, DISABLE_PROFILING);
    DPCPP_ENV_TYPE_DEF(lazy_reorder, LAZY_REORDER);
    DPCPP_ENV_TYPE_DEF(disable_tile_partition, DISABLE_TILE_PARTITION);
  } env;

  static auto _footer = []() -> bool {
    std::cerr << " *********************************************************/" << std::endl;
    return true;
  } ();

  switch (env_type) {
    case ENV_VERBOSE:
      return env.verbose;
    case ENV_FORCE_SYNC:
      return env.force_sync;
    case ENV_DISABLE_PROFILING:
      return env.disable_profiling;
    case ENV_LAZY_REORDER:
      return env.lazy_reorder;
    case ENV_DISABLE_TILE_PARTITION:
      return env.disable_tile_partition;
    default:
      return 0;
  }
}

int dpcpp_verbose() {
  return dpcpp_env(ENV_VERBOSE);
}

int dpcpp_force_sync() {
  return dpcpp_env(ENV_FORCE_SYNC);
}

bool dpcpp_profiling() {
  return !dpcpp_env(ENV_DISABLE_PROFILING);
}

int lazy_reorder_enabled() {
  return dpcpp_env(ENV_LAZY_REORDER);
}

bool disable_tile_partition() {
  return (bool)dpcpp_env(ENV_DISABLE_TILE_PARTITION);
}

bool onednn_layout_enabled() {
  return (bool)lazy_reorder_enabled();
}
