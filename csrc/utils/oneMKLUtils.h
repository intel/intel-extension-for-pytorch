#pragma once
#include <utils/DPCPP.h>

#define DPCPP_ONEMKL_SUBMIT(q, routine, ...) \
  { DPCPP_EXT_SUBMIT((q), "onemkl_kernel", routine(__VA_ARGS__)); }

namespace xpu {
namespace oneMKL {

// oneMKLExpInfo singleton
class oneMKLExpInfo {
 public:
  static oneMKLExpInfo& Instance(); // Singleton

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
      return onemkl_last_info;
    }
    return onemkl_last_info;
  }

 protected:
  oneMKLExpInfo() {
    onemkl_last_info = 0;
  }
  ~oneMKLExpInfo() {}
};

} // namespace oneMKL
} // namespace xpu
