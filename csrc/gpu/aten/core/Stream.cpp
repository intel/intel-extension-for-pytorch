#include <core/Stream.h>
#include <include/xpu/Stream.h>
#include <runtime/Device.h>
#include <runtime/Queue.h>
#include <utils/DPCPP.h>

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <utility>

namespace xpu {
namespace dpcpp {

std::ostream& operator<<(std::ostream& stream, const DPCPPStream& s) {
  return stream << s.unwrap();
}

static Queue* DPCPPStreamToQueue(DPCPPStream stream) {
  c10::DeviceIndex di = stream.device_index();
  QueueType st = queueType(static_cast<QueueId>(stream.unwrap().id()));
  size_t si = queueIdIndex(static_cast<QueueId>(stream.unwrap().id()));
  switch (st) {
    case QueueType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized stream ",
          stream.unwrap(),
          " (I think this should be the default stream, but I got a "
          "non-zero index ",
          si,
          ")");
      return getDefaultQueue(di);
    case QueueType::RESERVE:
      return getReservedQueue(di, si);
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream.unwrap(),
          " (I didn't recognize the stream type, ",
          std::to_string(static_cast<int>(st)),
          ")");
  }
}

DPCPPStream::DPCPPStream(Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == DeviceType::XPU);
}

DPCPPStream::DPCPPStream(Unchecked, Stream stream) : stream_(stream) {}

bool DPCPPStream::operator==(const DPCPPStream& other) const noexcept {
  return unwrap() == other.unwrap();
}

bool DPCPPStream::operator!=(const DPCPPStream& other) const noexcept {
  return unwrap() != other.unwrap();
}

DPCPPStream::operator Stream() const {
  return unwrap();
}

DeviceIndex DPCPPStream::device_index() const {
  return stream_.device_index();
}

Device DPCPPStream::device() const {
  return Device(DeviceType::XPU, device_index());
}

StreamId DPCPPStream::id() const {
  return stream_.id();
}

void DPCPPStream::synchronize() const {
  DeviceGuard guard{stream_.device()};
  auto queue = DPCPPStreamToQueue(*this);
  queue->getDpcppQueue().wait();
}

void DPCPPStream::synchronize_and_throw() const {
  DeviceGuard guard{stream_.device()};
  auto queue = DPCPPStreamToQueue(*this);
  queue->getDpcppQueue().wait_and_throw();
}

Stream DPCPPStream::unwrap() const {
  return stream_;
}

uint64_t DPCPPStream::pack() const noexcept {
  return stream_.pack();
}

void* DPCPPStream::opaque() const {
  auto queue = DPCPPStreamToQueue(*this);
  return reinterpret_cast<void*>(&queue->getDpcppQueue());
}

static DPCPPStream QueueToDPCPPStream(const Queue* ptr) {
  return DPCPPStream(
      DPCPPStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::XPU, ptr->getDeviceId()),
          getQueueId(ptr)));
}

DPCPPStream getDPCPPStreamFromPool(bool is_default, DeviceIndex device_index) {
  return QueueToDPCPPStream(getQueueFromPool(is_default, device_index));
}

DPCPPStream getDefaultDPCPPStream(DeviceIndex device_index) {
  return QueueToDPCPPStream(getDefaultQueue(device_index));
}

DPCPPStream getCurrentDPCPPStream(DeviceIndex device_index) {
  return QueueToDPCPPStream(getCurrentQueue(device_index));
}

void setCurrentDPCPPStream(DPCPPStream stream) {
  auto queue = DPCPPStreamToQueue(stream);
  setCurrentQueue(queue);
}

DPCPPStream getDPCPPStreamOnDevice(DeviceIndex device_index, int stream_index) {
  return QueueToDPCPPStream(getQueueOnDevice(device_index, stream_index));
}

} // namespace dpcpp

sycl::queue& get_queue_from_stream(c10::Stream stream) {
  dpcpp::DPCPPStream dpcpp_stream(stream);
  auto queue = dpcpp::DPCPPStreamToQueue(dpcpp_stream);
  return queue->getDpcppQueue();
}

} // namespace xpu
