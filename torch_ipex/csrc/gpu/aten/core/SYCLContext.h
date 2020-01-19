#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>

#include <core/SYCLStream.h>
#include <core/SYCLFunctions.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>

namespace at {
namespace sycl {

/*
 * A common SYCL interface for ATen.
 *
 * This interface is distinct from SYCLHooks, which defines an interface that links
 * to both CPU-only and SYCL builds. That interface is intended for runtime
 * dispatch and should be used from files that are included in both CPU-only and
 * SYCL builds.
 *
 * SYCLContext, on the other hand, should be preferred by files only included in
 * SYCL builds. It is intended to expose SYCL functionality in a consistent
 * manner.
 *
 * This means there is some overlap between the SYCLContext and SYCLHooks, but
 * the choice of which to use is simple: use SYCLContext when in a SYCL-only file,
 * use SYCLHooks otherwise.
 *
 * Note that SYCLContext simply defines an interface with no associated class.
 * It is expected that the modules whose functions compose this interface will
 * manage their own state. There is only a single SYCL context/state.
 */

/* Device info */
inline int64_t getNumGPUs() {
  return c10::sycl::device_count();
}

/**
 * In some situations, you may have compiled with SYCL, but no SYCL
 * device is actually available.  Test for this case using is_available().
 */
inline bool is_available() {
  int count;
  c10::sycl::syclGetDeviceCount(&count);
  return count > 0;
}

void createGlobalContext();
cl::sycl::context getGlobalContext();
CAFFE2_API Allocator* getSYCLDeviceAllocator();

} // namespace sycl
} // namespace at
