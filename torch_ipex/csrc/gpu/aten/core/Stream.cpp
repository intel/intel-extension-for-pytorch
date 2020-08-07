#include <c10/util/Exception.h>
#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Exception.h>
#include <core/Guard.h>
#include <core/Stream.h>
#include <utils/Env.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <cstdlib>

namespace at {
namespace dpcpp {

enum class StreamType : uint8_t {
  DEFAULT = 0x0,
  RESERVE = 0x1,
};

static constexpr int dpcppStreamPoolShift = 5;
static constexpr int dpcppStreamsPerPool = 32;

class DPCPPStreamImpl {
 public:
  DPCPPStreamImpl(
      DeviceIndex di,
      DPCPP::async_handler asyncHandler = dpcppAsyncHandler)
      : /* queue_(dpcppGetRawDevice(di), asyncHandler),*/
        queue_([&]() -> DPCPP::queue {
              return dpcpp_profiling() ?
                  DPCPP::queue(at::dpcpp::getGlobalContext(),
                   dpcppGetDeviceSelector(di),
                   asyncHandler,
#ifndef USE_USM
                   {DPCPP::property::queue::enable_profiling()}) :
#else
                   {DPCPP::property::queue::in_order(),
                    DPCPP::property::queue::enable_profiling()}) :
#endif
                  DPCPP::queue(at::dpcpp::getGlobalContext(),
                   dpcppGetDeviceSelector(di),
#ifndef USE_USM
                   asyncHandler);
#else
                   asyncHandler,
                   {DPCPP::property::queue::in_order()});
#endif
            } ()
        ), device_index_(di) {}

  DeviceIndex getDeviceIndex() const {
    return device_index_;
  };
  DPCPP::queue& get_dpcpp_queue() {
    return queue_;
  }
  ~DPCPPStreamImpl() = default;
  DPCPPStreamImpl() = default;
  C10_DISABLE_COPY_AND_ASSIGN(DPCPPStreamImpl);

 private:
  DPCPP::queue queue_;
  DeviceIndex device_index_;
};

static int dpcpp_num_devices;
static std::once_flag init_flag;
static std::vector<std::shared_ptr<DPCPPStreamImpl>> default_streams;
static std::vector<
    std::array<std::shared_ptr<DPCPPStreamImpl>, dpcppStreamsPerPool>>
    reserve_streams;
static std::deque<std::atomic<uint32_t>> reserve_counters;

std::ostream& operator<<(std::ostream& stream, StreamType s) {
  switch (s) {
    case StreamType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamType::RESERVE:
      stream << "RESERVE";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

static inline StreamType streamType(StreamId s) {
  return static_cast<StreamType>(s >> dpcppStreamPoolShift);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(s & ((1 << dpcppStreamPoolShift) - 1));
}

StreamId makeStreamId(StreamType st, size_t si) {
  return (static_cast<StreamId>(st) << dpcppStreamPoolShift) |
      static_cast<StreamId>(si);
}

static StreamId DPCPPStream_getStreamId(const DPCPPStreamImpl* ptr) {
  DeviceIndex di = ptr->getDeviceIndex();
  if (ptr == default_streams[di].get()) {
    return makeStreamId(StreamType::DEFAULT, 0);
  }

  for (int si = 0; si < reserve_streams[di].size(); si++) {
    if (ptr == reserve_streams[di][si].get()) {
      return makeStreamId(StreamType::RESERVE, si);
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

static thread_local DPCPPStreamImpl** current_streams = nullptr;

static void clearSyclStreams() {
  default_streams.clear();
  reserve_streams.clear();
}

static void initDPCPPStreams() {
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&dpcpp_num_devices));
  TORCH_CHECK(
      dpcpp_num_devices > 0,
      "Number of dpcpp devices should be greater than zero!");

  if (default_streams.size() == 0) {
    default_streams.resize(dpcpp_num_devices);
    reserve_counters.resize(dpcpp_num_devices);
    reserve_streams.resize(dpcpp_num_devices);
  }

  // init default streams
  for (int i = 0; i < dpcpp_num_devices; ++i) {
    default_streams[i] = std::make_shared<DPCPPStreamImpl>(i);
  }

  // init reserve stream pool
  for (int di = 0; di < dpcpp_num_devices; ++di) {
    for (auto pi = decltype(dpcppStreamsPerPool){0}; pi < dpcppStreamsPerPool;
         ++pi) {
      reserve_streams[di][pi] = std::make_shared<DPCPPStreamImpl>(di);
    }
    reserve_counters[di] = 0;
  }

  // Note: DPCPPRuntime's destruction happens before the destroy of the
  // global vars except the global vars with dpcpp type. This will make
  // our global device pool destruction crash. So we use atexit to
  // manually free all dpcpp streams. atexit callback happens before
  // DPCPPRuntime destruction.
  atexit(clearSyclStreams);
}

static void initDPCPPStreamsOnce() {
  std::call_once(init_flag, initDPCPPStreams);

  if (current_streams)
    return;

  current_streams =
      (DPCPPStreamImpl**)malloc(dpcpp_num_devices * sizeof(DPCPPStreamImpl*));
  for (int di = 0; di < dpcpp_num_devices; ++di) {
    current_streams[di] = default_streams[di].get();
  }
}

static inline void check_num_devices(DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < dpcpp_num_devices);
}

static uint32_t get_stream_index(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % dpcppStreamsPerPool;
}

DPCPPStreamImpl* DPCPPStreamToDPCPPStreamImpl(DPCPPStream stream) {
  c10::DeviceIndex di = stream.device_index();
  StreamType st = streamType(stream.unwrap().id());
  size_t si = streamIdIndex(stream.unwrap().id());
  switch (st) {
    case StreamType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized stream ",
          stream.unwrap(),
          " (I think this should be the default stream, but I got a "
          "non-zero index ",
          si,
          ")");
      return default_streams[di].get();
    case StreamType::RESERVE:
      return reserve_streams[di][si].get();
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream.unwrap(),
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

DPCPPStream DPCPPStreamImplToDPCPPStream(const DPCPPStreamImpl* ptr) {
  return DPCPPStream(
      DPCPPStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::DPCPP, ptr->getDeviceIndex()),
          DPCPPStream_getStreamId(ptr)));
}

DPCPP::queue& DPCPPStream::dpcpp_queue() const {
  auto streamImpl = DPCPPStreamToDPCPPStreamImpl(*this);
  return streamImpl->get_dpcpp_queue();
}

DPCPPStream getDPCPPStreamFromPool(
    const bool isDefault,
    DeviceIndex device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_num_devices(device_index);

  if (isDefault) {
    return getDefaultDPCPPStream(device_index);
  }

  const auto si = get_stream_index(reserve_counters[device_index]);
  return DPCPPStreamImplToDPCPPStream(reserve_streams[device_index][si].get());
}

DPCPPStream getDefaultDPCPPStream(DeviceIndex device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_num_devices(device_index);
  return DPCPPStreamImplToDPCPPStream(default_streams[device_index].get());
}

DPCPPStream getCurrentDPCPPStream(DeviceIndex device_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_num_devices(device_index);
  return DPCPPStreamImplToDPCPPStream(current_streams[device_index]);
}

void setCurrentDPCPPStream(DPCPPStream stream) {
  initDPCPPStreamsOnce();
  auto ptr = DPCPPStreamToDPCPPStreamImpl(stream);
  TORCH_INTERNAL_ASSERT(ptr);
  current_streams[ptr->getDeviceIndex()] = ptr;
}

DPCPPStream getDPCPPStreamOnDevice(DeviceIndex device_index, int stream_index) {
  initDPCPPStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_num_devices(device_index);
  if (stream_index == 0)
    return getDefaultDPCPPStream(device_index);
  TORCH_INTERNAL_ASSERT(stream_index <= dpcppStreamsPerPool);
  return DPCPPStreamImplToDPCPPStream(
      reserve_streams[device_index][stream_index - 1].get());
}

std::ostream& operator<<(std::ostream& stream, const DPCPPStream& s) {
  return stream << s.unwrap();
}

} // namespace dpcpp
} // namespace at
