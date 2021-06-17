#pragma once
#include <utils/DPCPP.h>

#ifdef BUILD_INTERNAL_DEBUG
#define DPCPP_ONEMKL_SUBMIT(q, routine, ...)                                  \
  {                                                                           \
    static auto verbose = dpcpp_verbose();                                    \
    if (verbose) {                                                            \
      IPEX_TIMER(t, verbose, __func__);                                       \
      auto e = routine(__VA_ARGS__);                                          \
      t.now("oneMKL submit");                                                 \
      e.wait_and_throw();                                                     \
      t.now("oneMKL event wait");                                             \
      DPCPP_Q_FORCE_SYNC(q);                                                  \
      t.now("oneMKL queue wait");                                             \
      dpcpp_log("onemkl_kernel", e);                                          \
    } else {                                                                  \
      auto e = routine(__VA_ARGS__);                                          \
      dpcpp_log("onemkl_kernel", e);                                          \
      e.wait_and_throw();                                                     \
      DPCPP_Q_FORCE_SYNC(q);                                                  \
    }                                                                         \
  }
#else
#define DPCPP_ONEMKL_SUBMIT(q, routine, ...)                                  \
  {                                                                           \
    static auto verbose = dpcpp_verbose();                                    \
    if (verbose) {                                                            \
      IPEX_TIMER(t, verbose, __func__);                                       \
      auto e = routine(__VA_ARGS__);                                          \
      t.now("oneMKL submit");                                                 \
      (q).throw_asynchronous();                                               \
      t.now("oneMKL throw asynchronous");                                     \
      DPCPP_Q_FORCE_SYNC(q);                                                  \
      t.now("oneMKL queue wait");                                             \
      dpcpp_log("onemkl_kernel", e);                                          \
    } else {                                                                  \
      auto e = routine(__VA_ARGS__);                                          \
      (q).throw_asynchronous();                                               \
      dpcpp_log("onemkl_kernel", e);                                          \
      DPCPP_Q_FORCE_SYNC(q);                                                  \
    }                                                                         \
  }
#endif

namespace xpu {
namespace oneMKL {

//OneMklExInfoManager singleton
class OneMklExInfoManager {
public:
  static OneMklExInfoManager& Instance() {
    static thread_local OneMklExInfoManager myInstance;
    return myInstance;
  }

  int64_t getLastInfo() {
    return accessLastInfo();
  }

  void setLastInfo(int64_t info) {
    accessLastInfo(true, info);
  }

private:
  int64_t onemkl_last_info;
  int64_t accessLastInfo(bool write = false, const int64_t info = 0) {
    if (true == write) {
      onemkl_last_info = info;
      return onemkl_last_info;;
    }
    return onemkl_last_info;
  }

protected:
  OneMklExInfoManager() {
    onemkl_last_info = 0;
  }
  ~OneMklExInfoManager() {}
};

} // namespace oneMKL
} // namespace xpu

