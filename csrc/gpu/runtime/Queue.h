#pragma once

#include <c10/core/Device.h>
#include <utils/DPCPP.h>

#include <cstdint>
#include <utility>

/*
 * Queue pool note.
 *
 * Currently, there is one pool without priority queue properties per device,
 * and a device's pool is lazily created. In XPU runtime, there is no
 * "default queue" semantics like the CUDA default stream.
 * TODO: We will integrate the priority queue to this pool in the future.
 *
 * There are 32 queues in this pool per device, and when a queue is requested
 * one of these queues is returned round-robin. That is, the first queue
 * requested is at index 0, the second at index 1... to index 31, then index 0
 * again.
 *
 * This means that if 33 queues are requested, the first and last queues
 * requested are actually the same queue (under the covers) and kernels enqueued
 * on them cannot run concurrently.
 *
 * Note: It is safe to enqueue a kernel on the same queue from two different
 * threads as the SYCL specification described.
 */

using namespace at;

namespace xpu {
namespace dpcpp {

// Put them here to share with DPCPPStream and oneDNN's runtime
static constexpr int kQueuesPerPoolBits = 5;
static constexpr int kQueuesPerPool = 1 << kQueuesPerPoolBits;
static constexpr int kQueueTypeBits = 3;

// Please keep synchronized with QueueIndex in aten/core/Stream.h
using QueueIndex = uint8_t;

enum class QueueType : uint8_t {
  UNUSED = 0x0,
  RESERVED = 0x1,
};

inline std::ostream& operator<<(std::ostream& stream, QueueType q) {
  switch (q) {
    case QueueType::UNUSED:
      stream << "UNUSED";
      break;
    case QueueType::RESERVED:
      stream << "RESERVED";
      break;
    default:
      stream << static_cast<uint8_t>(q);
      break;
  }
  return stream;
}

// Init queue pool's state.
void dpcppInitQueueStateOnce();

// Inits queue pool on the specified device.
void dpcppInitDeviceQueueOnce(DeviceIndex device_index);

// Helper to determine the index of the queue to return.
uint32_t dpcppGetQueueIndex(DeviceIndex device_index);

// Retrieve sycl queue reference from the queue pool.
sycl::queue& dpcppGetRawQueue(DeviceIndex device_index, QueueIndex queue_index);

} // namespace dpcpp
} // namespace xpu
