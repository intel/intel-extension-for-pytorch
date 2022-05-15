#pragma once

#include <runtime/Utils.h>
#include <utils/Helpers.h>
#include <utils/Settings.h>

namespace xpu {
namespace dpcpp {

#ifdef BUILD_SIMPLE_TRACE
class SimpleTrace {
 public:
  SimpleTrace(const char* name);
  ~SimpleTrace();

 private:
  static thread_local int gindent;
  static thread_local int gindex;

  int _index;
  std::string _pre_str;
  const char* _name;
  const bool _enabled;
};
#else
class SimpleTrace {
 public:
  SimpleTrace(const char* name) {}
};
#endif

} // namespace dpcpp
} // namespace xpu
