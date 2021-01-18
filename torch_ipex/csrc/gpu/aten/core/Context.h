#pragma once

#include <ATen/Context.h>
#include <ATen/core/ATenGeneral.h>

#include <core/DPCPPUtils.h>
#include <core/Functions.h>
#include <core/Memory.h>
#include <core/Stream.h>

namespace at {
namespace dpcpp {

/*
 * A common DPCPP interface for ATen.
 *
 * This interface is distinct from XPUHooks, which defines an interface that
 * links
 * to both CPU-only and DPCPP builds. That interface is intended for runtime
 * dispatch and should be used from files that are included in both CPU-only and
 * DPCPP builds.
 *
 * DPCPPContext, on the other hand, should be preferred by files only included
 * in
 * DPCPP builds. It is intended to expose DPCPP functionality in a consistent
 * manner.
 *
 * This means there is some overlap between the DPCPPContext and XPUHooks, but
 * the choice of which to use is simple: use DPCPPContext when in a DPCPP-only
 * file,
 * use XPUHooks otherwise.
 *
 * Note that DPCPPContext simply defines an interface with no associated class.
 * It is expected that the modules whose functions compose this interface will
 * manage their own state. There is only a single DPCPP context/state.
 */

/* Device info */
inline int64_t getNumGPUs() {
  return device_count();
}

/**
 * In some situations, you may have compiled with DPCPP, but no DPCPP
 * device is actually available.  Test for this case using is_available().
 */
inline bool is_available() {
  int count;
  dpcppGetDeviceCount(&count);
  return count > 0;
}

void createDeviceContext();
void clearDeviceContext();
DPCPP::context getDeviceContext(int device_index = 0);
CAFFE2_API Allocator* getDPCPPDeviceAllocator();

} // namespace dpcpp
} // namespace at
