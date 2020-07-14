#include <utils/Env.h>
#include <sstream>
#include <iostream>


int dpcpp_env(int env_type) {
  static struct {
    int level = [&]() -> int {
      auto env = std::getenv("IPEX_VERBOSE");
      int _level = 0;
      if (env) {
        _level = std::stoi(env, 0, 10);
      }
      std::cout << "IPEX-VERBOSE-LEVEL: " << _level << "\n";
      return _level;
    } ();

    int force_sync = [&]() -> int {
      auto env = std::getenv("FORCE_SYNC");
      int _force_sync = 0;
      if (env) {
        _force_sync = std::stoi(env, 0, 10);
      }
      std::cout << "Force SYNC: " << _force_sync << "\n";
      return _force_sync;
    } ();

    int lazy_reorder = [&]() -> int {
      auto env = std::getenv("LAZY_REORDER");
      int _lazy_reorder = 0;
      if (env) {
        _lazy_reorder = std::stoi(env, 0, 10);
      }
      std::cout << "Lazy Reorder: " << _lazy_reorder << "\n";
      return _lazy_reorder;
    } ();
  } env;

  switch (env_type) {
  case ENV_VERBOSE:
    return env.level;
  case ENV_FORCE_SYNC:
    return env.force_sync;
  case ENV_LAZY_REORDER:
    return env.lazy_reorder;
  default:
    return 0;
  }
}
