#include <profiler/XPUActivity.h>

#include <fmt/format.h>
#include <iostream>
#include <string>

#include <profiler/xpu_output_base.h>

namespace KINETO_NAMESPACE {

using namespace libkineto;

template <>
inline int64_t XPUActivity<pti_view_record_overhead>::timestamp() const {
  return nsToUs(unixEpochTimestamp(activity_._overhead_start_timestamp_ns));
}

template <>
inline int64_t XPUActivity<pti_view_record_overhead>::duration() const {
  return nsToUs(activity_._overhead_duration_ns);
}

template <>
inline int64_t XPUActivity<pti_view_record_kernel>::timestamp() const {
  return nsToUs(unixEpochTimestamp(activity_._start_timestamp));
}

template <>
inline int64_t XPUActivity<pti_view_record_kernel>::duration() const {
  return nsToUs(activity_._end_timestamp - activity_._start_timestamp);
}
template <class T>
inline int64_t XPUActivity<T>::timestamp() const {
  return nsToUs(unixEpochTimestamp(activity_._start_timestamp));
}

template <class T>
inline int64_t XPUActivity<T>::duration() const {
  return nsToUs(activity_._end_timestamp - activity_._start_timestamp);
}

template <>
inline const std::string GpuActivity<pti_view_record_kernel>::name() const {
  return raw()._name;
}

template <>
inline ActivityType GpuActivity<pti_view_record_kernel>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

template <class T>
inline void GpuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  return timestamp / 1000;
}

template <>
inline const std::string GpuActivity<pti_view_record_kernel>::metadataJson()
    const {
  const pti_view_record_kernel& kernel = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "appended": {}, "submitted": {}, "device": {}, "context": {},
      "queue": {}, "correlation": {})JSON",
      us(kernel._append_timestamp), us(kernel._submit_timestamp),
      XPUActivityApi::singleton().get_device_idx_from_uuid(kernel._device_uuid), 0,
      (int64_t)kernel._sycl_queue_id, kernel._correlation_id);
  // clang-format on
}

inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
  return fmt::format(
      "Memcpy {} ({} -> {})",
      ptiViewMemcpyTypeToString((pti_view_memcpy_type)kind),
      ptiViewMemoryTypeToString((pti_view_memory_type)src),
      ptiViewMemoryTypeToString((pti_view_memory_type)dst));
}

template <>
inline ActivityType GpuActivity<pti_view_record_memory_copy>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template <>
inline const std::string GpuActivity<pti_view_record_memory_copy>::name()
    const {
  return memcpyName(raw()._memcpy_type, raw()._mem_src, raw()._mem_dst);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template <>
inline const std::string GpuActivity<
    pti_view_record_memory_copy>::metadataJson() const {
  const pti_view_record_memory_copy& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      XPUActivityApi::singleton().get_device_idx_from_uuid(memcpy._device_uuid), (int64_t)memcpy._context_handle,
      (int64_t)memcpy._queue_handle, memcpy._correlation_id,
      memcpy._bytes, bandwidth(memcpy._bytes, memcpy._end_timestamp - memcpy._start_timestamp));
  // clang-format on
}

template <>
inline const std::string GpuActivity<pti_view_record_memory_fill>::name()
    const {
  std::string memory_kind =
      ptiViewMemoryTypeToString((pti_view_memory_type)raw()._mem_type);
  return fmt::format("Memset ({})", memory_kind);
}

template <>
inline ActivityType GpuActivity<pti_view_record_memory_fill>::type() const {
  return ActivityType::GPU_MEMSET;
}

template <>
inline const std::string GpuActivity<
    pti_view_record_memory_fill>::metadataJson() const {
  const pti_view_record_memory_fill& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      XPUActivityApi::singleton().get_device_idx_from_uuid(memset._device_uuid), (int64_t)memset._context_handle,
      (int64_t)memset._queue_handle, memset._correlation_id,
      memset._bytes, bandwidth(memset._bytes, memset._end_timestamp - memset._start_timestamp));
  // clang-format on
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void OverheadActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline bool OverheadActivity::flowStart() const {
  return false;
}

inline const std::string OverheadActivity::name() const {
  return ptiViewOverheadKindToString(activity_._overhead_kind);
}

inline const std::string OverheadActivity::metadataJson() const {
  return "";
}

inline bool RuntimeActivity::flowStart() const {
  std::string name(activity_._name);
  return std::find(
             correlate_runtime_ops.begin(),
             correlate_runtime_ops.end(),
             name) != correlate_runtime_ops.end();
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(
      R"JSON(
      "name": "{}", "correlation": {})JSON",
      activity_._name,
      activity_._correlation_id);
}

template <class T>
inline const std::string GpuActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
