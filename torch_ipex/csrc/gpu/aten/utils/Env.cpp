#include <utils/Env.h>
#include <sstream>
#include <iostream>


#define DPCPP_ENV_TYPE_DEF(type, var, log)          \
    int type = [&]() -> int {                       \
      auto env = std::getenv(#var);                 \
      int _##type = 0;                              \
      if (env) {                                    \
        _##type = std::stoi(env, 0, 10);            \
      }                                             \
      std::cout << #log << ": " << _##type << "\n"; \
      return _##type;                               \
    } ()

int dpcpp_env(int env_type) {
  static struct {
    DPCPP_ENV_TYPE_DEF(verbose, IPEX_VERBOSE, IPEX-VERBOSE-LEVEL);
    DPCPP_ENV_TYPE_DEF(force_sync, FORCE_SYNC, Force-SYNC);
    DPCPP_ENV_TYPE_DEF(lazy_reorder, LAZY_REORDER, Lazy-Reorder);
    DPCPP_ENV_TYPE_DEF(weight_opt, WEIGHT_OPT, Weight-OPT);
  } env;

  switch (env_type) {
  case ENV_VERBOSE:
    return env.verbose;
  case ENV_FORCE_SYNC:
    return env.force_sync;
  case ENV_LAZY_REORDER:
    return env.lazy_reorder;
  case ENV_WEIGHT_OPT:
    return env.weight_opt;
  default:
    return 0;
  }
}
