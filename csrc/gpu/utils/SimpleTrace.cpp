#ifdef BUILD_SIMPLE_TRACE

#include <utils/SimpleTrace.h>
#include <utils/SysUtil.h>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils/LogUtils.h"

#define LOG_LEVEL_TRACE 0

namespace torch_ipex::xpu {
namespace dpcpp {

int thread_local SimpleTrace::gindent = -1;
int thread_local SimpleTrace::gindex = -1;

SimpleTrace::SimpleTrace(const char* name)
    : _name(name), _enabled(Settings::I().get_log_level() >= LOG_LEVEL_TRACE) {
  if (_enabled) {
    gindent++;
    gindex++;

    _index = gindex;
    std::stringstream ss;
    ss << "Call  into  OP: " << _name << " (#" << _index << ")";
    std::string s = ss.str();

    IPEX_TRACE_LOG("OPS", "", s);
  }
}

SimpleTrace::~SimpleTrace() {
  if (_enabled) {
    if (Settings::I().is_sync_mode_enabled()) {
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      dpcpp_queue.wait();
    }
    std::stringstream ss;
    ss << "Step out of OP: " << _name << " (#" << _index << ")";
    IPEX_TRACE_LOG("OPS", "", ss.str());

    gindent--;
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu

#endif // BUILD_SIMPLE_TRACE
