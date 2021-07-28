#pragma once
#include <utils/DPCPP.h>

#define DPCPP_ONEMKL_SUBMIT(q, routine, ...)                      \
  {                                                               \
    DPCPP_EXT_SUBMIT((q), "onemkl_kernel", routine(__VA_ARGS__)); \
  }

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

