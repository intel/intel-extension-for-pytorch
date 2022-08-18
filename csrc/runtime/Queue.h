#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

using QueueId = int32_t;

enum class QueueType : uint8_t {
  DEFAULT = 0x0,
  RESERVE = 0x1,
};

class Queue {
 public:
  Queue(DeviceId di, sycl::async_handler asyncHandler = dpcppAsyncHandler)
      : queue_(std::make_unique<sycl::queue>(sycl::queue(
            dpcppGetDeviceContext(di),
            dpcppGetRawDevice(di),
            asyncHandler,
            {sycl::property::queue::in_order(),
             sycl::property::queue::enable_profiling()}))),
        device_id_(di) {}

  DeviceId getDeviceId() const {
    return device_id_;
  }

  sycl::queue& getDpcppQueue() {
    return *queue_;
  }

  ~Queue() = default;
  Queue() = default;
  IPEX_DISABLE_COPY_AND_ASSIGN(Queue);

 private:
  std::unique_ptr<sycl::queue> queue_;
  DeviceId device_id_;
};

QueueType queueType(QueueId s);

size_t queueIdIndex(QueueId s);

Queue* getDefaultQueue(DeviceId device_id = -1);

Queue* getReservedQueue(DeviceId device_id, QueueId queue_id);

Queue* getCurrentQueue(DeviceId device_id = -1);

void setCurrentQueue(Queue* queue);

Queue* getQueueFromPool(const bool isDefault, DeviceId device_id);

Queue* getQueueOnDevice(DeviceId device_id, QueueId queue_id);

QueueId getQueueId(const Queue* ptr);

DeviceId getDeviceIdOfCurrentQueue();

} // namespace dpcpp
} // namespace xpu
