#include <chrono>

namespace chrono = std::chrono;

namespace KINETO_NAMESPACE {

#ifdef _WIN32
uint64_t epochs_diff() {
  auto steady =
      chrono::time_point_cast<chrono::nanoseconds>(chrono::steady_clock::now());
  auto system =
      chrono::time_point_cast<chrono::nanoseconds>(chrono::system_clock::now());

  auto time_since_unix = system.time_since_epoch().count();
  auto time_since_boot = steady.time_since_epoch().count();
  return time_since_unix - time_since_boot;
}

uint64_t unixEpochTimestamp(uint64_t ts) {
  static uint64_t diff = epochs_diff();
  return ts + diff;
}
#else
uint64_t unixEpochTimestamp(uint64_t ts) {
  return ts;
}
#endif // _WIN32

} // namespace KINETO_NAMESPACE
