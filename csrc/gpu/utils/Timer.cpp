#include <utils/Timer.h>

#ifndef _MSC_VER
#include <sys/types.h>
#include <unistd.h>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif
#endif
#include <iomanip>
#include <sstream>
#include <string>

namespace xpu {
namespace dpcpp {

ipex_timer::ipex_timer(int verbose_level, std::string tag)
    : vlevel_(verbose_level) {
  if (verbose_level >= 1) {
    start_ = high_resolution_clock::now();
    tag_ = tag;
  }
}

void ipex_timer::now(std::string sstamp) {
  if (vlevel_ >= 1) {
    stamp_.push_back(high_resolution_clock::now());
    sstamp_.push_back(sstamp);
  }
}

void ipex_timer::event_duration(uint64_t us) {
  event_duration_.push_back(us);
}

ipex_timer::~ipex_timer() {
  if (vlevel_ >= 1) {
    auto pre = start_;
    auto end = high_resolution_clock::now();

    std::stringstream ps;

    ps << "[" << std::setfill(' ') << std::setw(13)
       << std::to_string(getpid()) + "." + std::to_string(gettid()) << "]  "
       << tag_ << ":";
    for (int i = 0; i < stamp_.size(); i++) {
      auto stamp = stamp_.at(i);
      auto sstamp = sstamp_.at(i);
      ps << " " << sstamp << " = "
         << duration_cast<microseconds>(stamp - pre).count() << " us,";
      pre = stamp;
    }
    for (int j = 0; j < event_duration_.size(); j++) {
      ps << " event_" << j << " duration"
         << " = " << event_duration_.at(j) << " us,";
    }
    ps << " total = " << duration_cast<microseconds>(end - start_).count()
       << " us" << std::endl;

    std::cout << ps.str();
    fflush(stdout);
  }
}

} // namespace dpcpp
} // namespace xpu
