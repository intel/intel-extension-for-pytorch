#include <runtime/Queue.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include <utils/Settings.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace xpu {
namespace dpcpp {

static int dpcpp_num_devices;
static std::once_flag init_flag;
static std::vector<std::shared_ptr<Queue>> default_queues;
static std::vector<std::array<std::shared_ptr<Queue>, QueuePerPool>>
    reserved_queues;
static std::deque<std::atomic<uint32_t>> reserve_counters;

QueueType queueType(QueueId s) {
  return static_cast<QueueType>(s >> QueuePoolShift);
}

size_t queueIdIndex(QueueId s) {
  return static_cast<size_t>(s & ((1 << QueuePoolShift) - 1));
}

QueueId makeQueueId(QueueType st, size_t queue_id) {
  return (static_cast<QueueId>(st) << QueuePoolShift) |
      static_cast<QueueId>(queue_id);
}

QueueId getQueueId(const Queue* ptr) {
  auto di = ptr->getDeviceId();
  if (ptr == default_queues[di].get()) {
    return makeQueueId(QueueType::DEFAULT, 0);
  }

  for (int queue_id = 0; queue_id < reserved_queues[di].size(); queue_id++) {
    if (ptr == reserved_queues[di][queue_id].get()) {
      return makeQueueId(QueueType::RESERVE, queue_id);
    }
  }

  TORCH_INTERNAL_ASSERT(
      0,
      "Could not compute stream ID for ",
      ptr,
      " on device ",
      di,
      " (something has gone horribly wrong!)");
}

static thread_local Queue** current_queues = nullptr;

static void clearQueues() {
  default_queues.clear();
  reserved_queues.clear();
  if (current_queues) {
    free(current_queues);
  }
}

static void initQueuePool() {
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&dpcpp_num_devices));
  TORCH_CHECK(
      dpcpp_num_devices > 0,
      "Number of dpcpp devices should be greater than zero!");

  if (default_queues.size() == 0) {
    default_queues.resize(dpcpp_num_devices);
    reserve_counters.resize(dpcpp_num_devices);
    reserved_queues.resize(dpcpp_num_devices);
  }

  // init default queues
  for (int i = 0; i < dpcpp_num_devices; ++i) {
    default_queues[i] = std::make_shared<Queue>(i);
  }

  // init reserve queue pool
  for (int di = 0; di < dpcpp_num_devices; ++di) {
    for (auto pi = decltype(QueuePerPool){0}; pi < QueuePerPool; ++pi) {
      reserved_queues[di][pi] = std::make_shared<Queue>(di);
    }
    reserve_counters[di] = 0;
  }

  // Note: DPCPPRuntime's destruction happens before the destroy of the
  // global vars except the global vars with dpcpp type. This will make
  // our global device pool destruction crash. So we use atexit to
  // manually free all dpcpp queues. atexit callback happens before
  // DPCPPRuntime destruction.
  atexit(clearQueues);
}

static void initQueuePoolOnce() {
  std::call_once(init_flag, initQueuePool);

  if (current_queues) {
    return;
  }

  current_queues = (Queue**)malloc(dpcpp_num_devices * sizeof(Queue*));
  for (int di = 0; di < dpcpp_num_devices; ++di) {
    current_queues[di] = default_queues[di].get();
  }
}

static inline void check_num_devices(DeviceId device_id) {
  TORCH_INTERNAL_ASSERT(device_id >= 0 && device_id < dpcpp_num_devices);
}

static uint32_t get_queue_index(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % QueuePerPool;
}

Queue* getCurrentQueue(DeviceId device_id) {
  initQueuePoolOnce();
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  check_num_devices(device_id);
  return current_queues[device_id];
}

void setCurrentQueue(Queue* queue) {
  initQueuePoolOnce();
  TORCH_INTERNAL_ASSERT(queue);
  current_queues[queue->getDeviceId()] = queue;
}

Queue* getDefaultQueue(DeviceId device_id) {
  initQueuePoolOnce();
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  check_num_devices(device_id);
  return default_queues[device_id].get();
}

Queue* getReservedQueue(DeviceId device_id, QueueId queue_id) {
  initQueuePoolOnce();
  return reserved_queues[device_id][queue_id].get();
}

Queue* getQueueFromPool(const bool isDefault, DeviceId device_id) {
  initQueuePoolOnce();
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  check_num_devices(device_id);

  if (isDefault) {
    return getDefaultQueue(device_id);
  }

  const auto queue_id = get_queue_index(reserve_counters[device_id]);
  return getReservedQueue(device_id, queue_id);
}

Queue* getQueueOnDevice(DeviceId device_id, QueueId queue_id) {
  initQueuePoolOnce();
  if (device_id == -1) {
    AT_DPCPP_CHECK(dpcppGetDevice(&device_id));
  }
  check_num_devices(device_id);

  if (queue_id == 0) {
    return getDefaultQueue(device_id);
  }
  TORCH_INTERNAL_ASSERT(queue_id <= QueuePerPool);
  return reserved_queues[device_id][queue_id - 1].get();
}

DeviceId getDeviceIdOfCurrentQueue() {
  return getCurrentQueue()->getDeviceId();
}

} // namespace dpcpp
} // namespace xpu
