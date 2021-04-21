#include <utils/Env.h>
#include <sstream>
#include <iostream>

/*
 * All available launch options for IPEX
 * IPEX_VERBOSE:            Default = 0, Set verbose level in IPEX
 * IPEX_FORCE_SYNC:         Default = 0, Set 1 to enforce blocked/sync execution mode
 * IPEX_DISABLE_PROFILING:  Default = 0, Set 1 to disable IPEX event profiling
 * IPEX_LAZY_REORDER:       Default = 0, Set 1 to enable lazy reorder to avoid unnecessary reorders
 * IPEX_WEIGHT_CACHE:       Default = 0, Set 1 to cache the packed weight in original weight Tensor
 * IPEX_TILE_AS_DEVICE:     Default = 0, Set 1 to map every device to physical tile.
*/
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
    DPCPP_ENV_TYPE_DEF(weight_cache, WEIGHT_CACHE);
    DPCPP_ENV_TYPE_DEF(tile_as_device, TILE_AS_DEVICE);
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
    case ENV_WEIGHT_CACHE:
      return env.weight_cache;
    case ENV_TILE_AS_DEVICE:
      return env.tile_as_device;
    default:
      return 0;
  }
}
