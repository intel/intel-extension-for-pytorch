#ifdef BUILD_SIMPLE_TRACE

#ifndef _MSC_VER
#include <sys/types.h>
#include <unistd.h>
#endif
#include <utils/SimpleTrace.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace xpu {
namespace dpcpp {

int thread_local SimpleTrace::gindent = -1;
int thread_local SimpleTrace::gindex = -1;

SimpleTrace::SimpleTrace(const char* name)
    : _name(name), _enabled(Settings::I().is_simple_trace_enabled()) {
  if (_enabled) {
    gindent++;
    gindex++;

    std::stringstream ps;
    ps << "[" << std::setfill(' ') << std::setw(13)
       << std::to_string(getpid()) + "." + std::to_string(gettid()) << "] "
       << std::setw(gindent * 2 + 1) << " ";
    _pre_str = ps.str();

    _index = gindex;
    std::cout << _pre_str << "Call  into  OP: " << _name << " (#" << _index
              << ")" << std::endl;
    fflush(stdout);
  }
}

SimpleTrace::~SimpleTrace() {
  if (_enabled) {
    if (Settings::I().is_sync_mode_enabled()) {
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      dpcpp_queue.wait();
    }
    std::cout << _pre_str << "Step out of OP: " << _name << " (#" << _index
              << ")" << std::endl;
    fflush(stdout);
    gindent--;
  }
}

} // namespace dpcpp
} // namespace xpu

#endif // BUILD_SIMPLE_TRACE
