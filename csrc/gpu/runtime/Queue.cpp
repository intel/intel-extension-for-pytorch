#include <c10/util/Exception.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <runtime/Queue.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include <iostream>

namespace xpu {
namespace dpcpp {
namespace {

// Global queue state and constants
static std::once_flag init_flag;

// Note: the number of XPU devices is determined at run time, and the sycl queue
// pool is lazily initialized when the first queue is requested for a device.
// The device flags track the initialization of each device, while the reserved
// counters track, for each device, the next queue in the pool to be returned
// when a queue is requested (round-robin fashion, see the note in Stream.h).
// The queues are "leaked": SYCL runtime's destruction happens before the
// destroy of the global variables and thus this will make our global pool
// destruction crash. It's likely an issue in SYCL, but to be safe - let's just
// "forget" the destruction.
static std::deque<std::once_flag> device_flags;
static std::vector<std::array<std::unique_ptr<sycl::queue>, kQueuesPerPool>>
    reserved_queues;
static std::deque<std::atomic<uint32_t>> reserved_counters;

} // anonymous namespace

// Warning: this function must only be called once!
static void initGlobalQueueState() {
  int num_devices;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&num_devices));
  TORCH_CHECK(
      num_devices > 0, "Number of XPU devices should be greater than zero!");

  device_flags.resize(num_devices);
  reserved_queues.resize(num_devices);
  reserved_counters.resize(num_devices);
}

// Warning: only call once per device!
static void initDeviceQueue(DeviceId device_index) {
  // Creates the reserved sycl queue pools for the specified device.
  for (auto i = 0; i < kQueuesPerPool; i++) {
    reserved_queues[device_index][i] =
        std::make_unique<sycl::queue>(sycl::queue(
            dpcppGetDeviceContext(device_index),
            dpcppGetRawDevice(device_index),
            dpcppAsyncHandler,
            {sycl::property::queue::in_order(),
             sycl::property::queue::enable_profiling()}));
  }

  reserved_counters[device_index] = 0;
}

// Inits queue pool's size to ensure initialization only occurs once.
void dpcppInitQueueStateOnce() {
  std::call_once(init_flag, initGlobalQueueState);
}

// Creates the reserved sycl queue pools for the specified device to ensure
// initialization only occurs once.
void dpcppInitDeviceQueueOnce(DeviceId device_index) {
  std::call_once(device_flags[device_index], initDeviceQueue, device_index);
}

// Helper to determine the index of the queue to return.
// Note: Queues are returned round-robin (see note in Queue.h)
uint32_t dpcppGetQueueIndex(DeviceId device_index) {
  auto raw_idx = reserved_counters[device_index]++;
  return raw_idx % kQueuesPerPool;
}

// Return sycl queue reference using DeviceId and QueueIndex that can retrieve
// the corresponding queue from the pool.
sycl::queue& dpcppGetRawQueue(DeviceId device_index, QueueIndex queue_index) {
  return *reserved_queues[device_index][queue_index];
}

} // namespace dpcpp
} // namespace xpu
