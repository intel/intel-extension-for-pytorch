#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/core/DeviceGuard.h>

#include <runtime/DPCPP.h>
#include <runtime/Device.h>
#include <runtime/Macros.h>
#include <runtime/Exception.h>


using namespace at;
using QueueId = c10::StreamId;

namespace xpu {
namespace dpcpp {

enum class QueueType : uint8_t {
  DEFAULT = 0x0,
  RESERVE = 0x1,
};

class Queue {
 public:
  Queue(
      DeviceIndex di,
      DPCPP::async_handler asyncHandler = dpcppAsyncHandler)
      : queue_([&]() -> DPCPP::queue {
              return dpcpp_profiling() ?
                  DPCPP::queue(dpcppGetRawDevice(di), asyncHandler,
                   {DPCPP::property::queue::in_order(),
                    DPCPP::property::queue::enable_profiling()})
                  : DPCPP::queue(dpcppGetRawDevice(di), asyncHandler,
                      {DPCPP::property::queue::in_order()});
            } ()
        ), device_index_(di) {}

  DeviceIndex getDeviceIndex() const {
    return device_index_;
  }

  DPCPP::queue& getDpcppQueue() {
    return queue_;
  }

  ~Queue() = default;
  Queue() = default;
  IPEX_DISABLE_COPY_AND_ASSIGN(Queue);

 private:
  DPCPP::queue queue_;
  DeviceIndex device_index_;
};

QueueType queueType(QueueId s);

size_t queueIdIndex(QueueId s);

Queue* getDefaultQueue(DeviceIndex device_index);

Queue* getReservedQueue(DeviceIndex device_index, QueueId queue_id);

Queue* getCurrentQueue(DeviceIndex device_index);

void setCurrentQueue(Queue* queue);

Queue* getQueueFromPool(const bool isDefault, DeviceIndex device_index);

Queue* getQueueOnDevice(DeviceIndex device_index, QueueId queue_id);

QueueId getQueueId(const Queue* ptr);

} // namespace dpcpp
} // namespace xpu
