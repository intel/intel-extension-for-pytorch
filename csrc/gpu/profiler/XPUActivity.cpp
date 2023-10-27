#include "XPUActivity.h"

#include <fmt/format.h>
#include <iostream>

#include "onepti_activity_api.h"
#include "xpu_output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

template <>
inline const std::string GpuActivity<Onepti_ActivityKernel>::name() const {
  return raw().name;
}

template <>
inline ActivityType GpuActivity<Onepti_ActivityKernel>::type() const {
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
inline const std::string GpuActivity<Onepti_ActivityKernel>::metadataJson()
    const {
  const Onepti_ActivityKernel& kernel = raw();
  // clang-format off
//   return fmt::format(R"JSON(
//       "queued": {}, "device": {}, "context": {},
//       "stream": {}, "correlation": {},
//       "registers per thread": {},
//       "shared memory": {},
//       "blocks per SM": {},
//       "warps per SM": {},
//       "grid": [{}, {}, {}],
//       "block": [{}, {}, {}],
//       "est. achieved occupancy %": {})JSON",
//       us(kernel.queued), kernel.deviceId, kernel.contextId,
//       kernel.streamId, kernel.correlationId,
//       kernel.registersPerThread,
//       kernel.staticSharedMemory + kernel.dynamicSharedMemory,
//       blocksPerSm(kernel),
//       warpsPerSm(kernel),
//       kernel.gridX, kernel.gridY, kernel.gridZ,
//       kernel.blockX, kernel.blockY, kernel.blockZ,
//       (int) (0.5 + kernelOccupancy(kernel) * 100.0));
  return fmt::format(R"JSON(
      "appended": {}, "submitted": {},
      "device": {}, "queue": {}, "correlation": {})JSON",
      us(kernel.appended), us(kernel.submitted),
      kernel.deviceId, kernel.queueId, kernel.correlationId);
  // clang-format on
}

// inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
//   return fmt::format(
//       "Memcpy {} ({} -> {})",
//       memcpyKindString((CUpti_ActivityMemcpyKind)kind),
//       memoryKindString((CUpti_ActivityMemoryKind)src),
//       memoryKindString((CUpti_ActivityMemoryKind)dst));
// }
//
// template<>
// inline ActivityType GpuActivity<CUpti_ActivityMemcpy>::type() const {
//   return ActivityType::GPU_MEMCPY;
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemcpy>::name() const {
//   return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
// }
//
// inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
//   return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 /
//   duration);
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemcpy>::metadataJson()
// const {
//   const CUpti_ActivityMemcpy& memcpy = raw();
//   // clang-format off
//   return fmt::format(R"JSON(
//       "device": {}, "context": {},
//       "stream": {}, "correlation": {},
//       "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
//       memcpy.deviceId, memcpy.contextId,
//       memcpy.streamId, memcpy.correlationId,
//       memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
//   // clang-format on
// }
//
//
// template<>
// inline ActivityType GpuActivity<CUpti_ActivityMemcpy2>::type() const {
//   return ActivityType::GPU_MEMCPY;
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemcpy2>::name() const {
//   return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemcpy2>::metadataJson()
// const {
//   const CUpti_ActivityMemcpy2& memcpy = raw();
//   // clang-format off
//   return fmt::format(R"JSON(
//       "fromDevice": {}, "inDevice": {}, "toDevice": {},
//       "fromContext": {}, "inContext": {}, "toContext": {},
//       "stream": {}, "correlation": {},
//       "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
//       memcpy.srcDeviceId, memcpy.deviceId, memcpy.dstDeviceId,
//       memcpy.srcContextId, memcpy.contextId, memcpy.dstContextId,
//       memcpy.streamId, memcpy.correlationId,
//       memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
//   // clang-format on
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemset>::name() const {
//   const char* memory_kind =
//     memoryKindString((CUpti_ActivityMemoryKind)raw().memoryKind);
//   return fmt::format("Memset ({})", memory_kind);
// }
//
// template<>
// inline ActivityType GpuActivity<CUpti_ActivityMemset>::type() const {
//   return ActivityType::GPU_MEMSET;
// }
//
// template<>
// inline const std::string GpuActivity<CUpti_ActivityMemset>::metadataJson()
// const {
//   const CUpti_ActivityMemset& memset = raw();
//   // clang-format off
//   return fmt::format(R"JSON(
//       "device": {}, "context": {},
//       "stream": {}, "correlation": {},
//       "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
//       memset.deviceId, memset.contextId,
//       memset.streamId, memset.correlationId,
//       memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
//   // clang-format on
// }

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

// inline void OverheadActivity::log(ActivityLogger& logger) const {
//   logger.handleActivity(*this);
// }
//
// inline bool OverheadActivity::flowStart() const {
//   return false;
// }
//
// inline const std::string OverheadActivity::metadataJson() const {
//   return "";
// }

inline bool RuntimeActivity::flowStart() const {
  //   return activity_.cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
  //   ||
  //       (activity_.cbid >= CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 &&
  //        activity_.cbid <= CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020)
  //        ||
  //       activity_.cbid ==
  //           CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
  //       activity_.cbid ==
  //           CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000
  //           ||
  //       activity_.cbid ==
  //           CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000;
  std::string name(activity_.name);
  return name == "zeCommandListAppendLaunchKernel";
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(
      R"JSON(
      "name": "{}", "correlation": {})JSON",
      activity_.name,
      activity_.correlationId);
}

template <class T>
inline const std::string GpuActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
