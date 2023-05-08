#include <c10/util/Exception.h>
#include <core/Device.h>
#include <core/Stream.h>
#include <runtime/Device.h>
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

// Global stream state and constants
static DeviceId num_devices = -1;

// Thread-local current streams, it stores StreamId that can calculate
// QueueIndex that can retrieve the current queue from queue pool.
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
// zeros            queue index      QueueType
//
// Where QueueType:
//  000 = UNUSED
//  001 = reserved queue
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on QueueIndex and QueueType being non-negative;

static inline QueueType queueType(StreamId s) {
  int mask_for_type = (1 << kQueueTypeBits) - 1;
  return static_cast<QueueType>(s & mask_for_type);
}

static inline QueueIndex queueIndex(StreamId s) {
  return static_cast<QueueIndex>(
      (s >> kQueueTypeBits) & ((1 << kQueuesPerPoolBits) - 1));
}

static inline StreamId makeStreamId(QueueType qt, QueueIndex qi) {
  return (static_cast<StreamId>(qi) << kQueueTypeBits) |
      static_cast<StreamId>(qt);
}

// Init queue pool's state and current streams' state
static void initDPCPPStreamsOnce() {
  dpcppInitQueueStateOnce();

  if (current_streams) {
    return;
  }

  num_devices = xpu::dpcpp::device_count();
  TORCH_CHECK(
      num_devices > 0, "Number of XPU devices should be greater than zero!");

  // Inits current streams (thread local) to the first queue in the queue pool.
  // Note: the queue pools have not been initialized yet. They will be
  // initialized in dpcppInitDeviceQueueOnce for the specified device.
  current_streams = std::make_unique<StreamId[]>(num_devices);
  for (auto i = 0; i < num_devices; i++) {
    current_streams[i] = makeStreamId(QueueType::RESERVED, 0);
  }
}

// Helper to verify the device index is valid.
static inline void check_device_index(DeviceId device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_devices);
}

DPCPPStream DPCPPStreamForId(DeviceId device_index, StreamId stream_id) {
  return DPCPPStream(
      DPCPPStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::XPU, device_index),
          stream_id));
}

} // anonymous namespace

void DPCPPStream::synchronize() const {
  DeviceGuard guard{stream_.device()};
  dpcppGetRawQueue(device_index(), queue_index()).wait();
}

void DPCPPStream::synchronize_and_throw() const {
  DeviceGuard guard{stream_.device()};
  dpcppGetRawQueue(device_index(), queue_index()).wait_and_throw();
}

// See Note [StreamId assignment]
void* DPCPPStream::queue() const {
  DeviceId device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  QueueType qt = queueType(stream_id);
  QueueIndex qi = queueIndex(stream_id);
  switch (qt) {
    case QueueType::UNUSED:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized queue ",
          stream_,
          " (I didn't recognize the queue type, ",
          qt,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that;");
    case QueueType::RESERVED:
      return reinterpret_cast<void*>(&dpcppGetRawQueue(device_index, qi));
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized queue ",
          stream_,
          " (I didn't recognize the queue type, ",
          qt,
          ")");
  }
}

// Returns a sycl queue index in queue pool.
QueueIndex DPCPPStream::queue_index() const {
  return queueIndex(stream_.id());
}

// Returns a stream from the requested pool.
// Note: when called the first time on a device, this will create the queue pool
// for that device.
DPCPPStream getStreamFromPool(
    const bool isHighPriority,
    DeviceId device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = xpu::dpcpp::current_device();
  check_device_index(device_index);
  dpcppInitDeviceQueueOnce(device_index);
  return DPCPPStreamForId(
      device_index,
      makeStreamId(QueueType::RESERVED, dpcppGetQueueIndex(device_index)));
}

// Note: when called the first time on a device, this will create the queue pool
// for that device.
DPCPPStream getCurrentDPCPPStream(DeviceId device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = xpu::dpcpp::current_device();
  check_device_index(device_index);
  dpcppInitDeviceQueueOnce(device_index);
  return DPCPPStreamForId(device_index, current_streams[device_index]);
}

void setCurrentDPCPPStream(DPCPPStream stream) {
  initDPCPPStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const DPCPPStream& s) {
  return stream << s.unwrap();
}

void deviceSynchronize(DeviceIndex device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = xpu::dpcpp::current_device();
  check_device_index(device_index);
  dpcppInitDeviceQueueOnce(device_index);

  // For each device, we have 32 (kQueuesPerPool) reserved queues.
  for (auto i = 0; i < kQueuesPerPool; i++) {
    /**
     * Why we don NOT need a barrier for synchronization snapshot here? Because
     * xpu::dpcpp::queue_barrier is so much time-consuming.
     */
    dpcppGetRawQueue(device_index, i).wait();
  }
}

} // namespace dpcpp

sycl::queue& get_queue_from_stream(c10::Stream stream) {
  dpcpp::DPCPPStream dpcpp_stream(stream);
  return dpcppGetRawQueue(
      dpcpp_stream.device_index(), dpcpp_stream.queue_index());
}

} // namespace xpu
