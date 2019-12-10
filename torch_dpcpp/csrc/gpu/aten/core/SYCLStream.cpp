#include <c10/dpcpp/SYCLStream.h>
#include <c10/dpcpp/SYCLGuard.h>
#include <c10/dpcpp/SYCLException.h>
#include <c10/util/Exception.h>

#include <mutex>
#include <atomic>
#include <cstdint>
#include <deque>
#include <vector>
#include <array>
#include <memory>

#include <cstdlib>

namespace c10 {
namespace sycl {

enum class StreamType: uint8_t {
  DEFAULT = 0x0,
  RESERVE = 0x1,
};

static constexpr int syclStreamPoolShift = 5;
static constexpr int syclStreamsPerPool = 32;

static cl::sycl::async_handler syclAsyncHandler = [](cl::sycl::exception_list eL) {
  for (auto& e : eL) {
    C10_SYCL_TRY
    std::rethrow_exception(e);
    C10_SYCL_CATCH_RETHROW(__FILE__, __LINE__)
  }
};

class SYCLStreamImpl {
public:
  SYCLStreamImpl(DeviceIndex di, cl::sycl::async_handler asyncHandler = syclAsyncHandler):
      queue_(syclGetRawDevice(di), asyncHandler), device_index_(di) {};
  DeviceIndex getDeviceIndex() const { return device_index_; };
  cl::sycl::queue& get_sycl_queue() { return queue_; }
  ~SYCLStreamImpl() = default;
  SYCLStreamImpl() = default;
  C10_DISABLE_COPY_AND_ASSIGN(SYCLStreamImpl);
private:
  cl::sycl::queue   queue_;
  DeviceIndex       device_index_;
};

static int sycl_num_devices;
static std::once_flag init_flag;
static std::vector<std::shared_ptr<SYCLStreamImpl> > default_streams;
static std::vector<std::array<std::shared_ptr<SYCLStreamImpl>, syclStreamsPerPool>> reserve_streams;
static std::deque<std::atomic<uint32_t> > reserve_counters;

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
  return static_cast<StreamType>(s >> syclStreamPoolShift);
}

static inline size_t streamIdIndex(StreamId s) {
    return static_cast<size_t>(s & ((1 << syclStreamPoolShift) - 1));
}

StreamId makeStreamId(StreamType st, size_t si) {
  return (static_cast<StreamId>(st) << syclStreamPoolShift) | static_cast<StreamId>(si);
}

static StreamId SYCLStream_getStreamId(const SYCLStreamImpl *ptr) {
  DeviceIndex di = ptr->getDeviceIndex();
  if (ptr == default_streams[di].get()) {
    return makeStreamId(StreamType::DEFAULT, 0);
  }

  for (int si = 0; si < reserve_streams[di].size(); si++) {
    if (ptr == reserve_streams[di][si].get()) {
      return makeStreamId(StreamType::RESERVE, si);
    }
  }

  AT_ASSERTM(0, "Could not compute stream ID for ", ptr, " on device ", di,
                " (something has gone horribly wrong!)");
}

static thread_local SYCLStreamImpl** current_streams = nullptr;

static void clearSyclStreams() {
  default_streams.clear();
  reserve_streams.clear();
}

static void initSYCLStreams() {
  C10_SYCL_CHECK(syclGetDeviceCount(&sycl_num_devices));
  TORCH_CHECK(sycl_num_devices > 0, "Number of sycl devices should be greater than zero!");

  if (default_streams.size() == 0) {
    default_streams.resize(sycl_num_devices);
    reserve_counters.resize(sycl_num_devices);
    reserve_streams.resize(sycl_num_devices);
  }

  // init default streams
  for (int i = 0; i < sycl_num_devices; ++i) {
    default_streams[i] = std::make_shared<SYCLStreamImpl>(i);
  }

  // init reserve stream pool
  for (int di = 0; di < sycl_num_devices; ++di) {
    for (auto pi = decltype(syclStreamsPerPool){0}; pi < syclStreamsPerPool; ++pi) {
      reserve_streams[di][pi] = std::make_shared<SYCLStreamImpl>(di);
    }
    reserve_counters[di] = 0;
  }

  // Note: SYCLRuntime's destruction happens before the destroy of the
  // global vars except the global vars with sycl type. This will make
  // our global device pool destruction crash. So we use atexit to
  // manually free all sycl streams. atexit callback happens before
  // SYCLRuntime destruction.
  atexit(clearSyclStreams);
}

static void initSYCLStreamsOnce() {
  std::call_once(init_flag, initSYCLStreams);

  if (current_streams) return;

  current_streams = (SYCLStreamImpl**)malloc(sycl_num_devices * sizeof(SYCLStreamImpl*));
  for (int di = 0; di < sycl_num_devices; ++di) {
    current_streams[di] = default_streams[di].get();
  }
}

static inline void check_num_devices(DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < sycl_num_devices);
}

static uint32_t get_stream_index(std::atomic<uint32_t> &counter) {
  auto raw_idx = counter++;
  return raw_idx % syclStreamsPerPool;
}

SYCLStreamImpl* SYCLStreamToSYCLStreamImpl(SYCLStream stream) {
  c10::DeviceIndex di = stream.device_index();
  StreamType st = streamType(stream.unwrap().id());
  size_t si = streamIdIndex(stream.unwrap().id());
  switch(st) {
    case StreamType::DEFAULT:
      AT_ASSERTM(si == 0, "Unrecognized stream ", stream.unwrap(),
                " (I think this should be the default stream, but I got a non-zero index ", si, ")");
      return default_streams[di].get();
    case StreamType::RESERVE:
      return reserve_streams[di][si].get();
    default:
      AT_ASSERTM(0, "Unrecognized stream ", stream.unwrap(), " (I didn't recognize the stream type, ", st, ")");
  }
}

SYCLStream SYCLStreamImplToSYCLStream(const SYCLStreamImpl *ptr) {
  return SYCLStream(SYCLStream::UNCHECKED,
              Stream(Stream::UNSAFE,
                     c10::Device(DeviceType::SYCL, ptr->getDeviceIndex()),
                     SYCLStream_getStreamId(ptr)));
}

cl::sycl::queue& SYCLStream::sycl_queue() const {
  auto streamImpl = SYCLStreamToSYCLStreamImpl(*this);
  return streamImpl->get_sycl_queue();
}

SYCLStream getSYCLStreamFromPool(const bool isDefault, DeviceIndex device_index) {
  initSYCLStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_num_devices(device_index);

  if (isDefault) {
    return getDefaultSYCLStream(device_index);
  }

  const auto si = get_stream_index(reserve_counters[device_index]);
  return SYCLStreamImplToSYCLStream(reserve_streams[device_index][si].get());
}

SYCLStream getDefaultSYCLStream(DeviceIndex device_index) {
  initSYCLStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_num_devices(device_index);
  return SYCLStreamImplToSYCLStream(default_streams[device_index].get());
}

SYCLStream getCurrentSYCLStream(DeviceIndex device_index) {
  initSYCLStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_num_devices(device_index);
  return SYCLStreamImplToSYCLStream(current_streams[device_index]);
}

void setCurrentSYCLStream(SYCLStream stream) {
  initSYCLStreamsOnce();
  auto ptr = SYCLStreamToSYCLStreamImpl(stream);
  AT_ASSERT(ptr);
  current_streams[ptr->getDeviceIndex()] = ptr;
}

SYCLStream getSYCLStreamOnDevice(DeviceIndex device_index, int stream_index) {
  initSYCLStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_num_devices(device_index);
  if (stream_index == 0) return getDefaultSYCLStream(device_index);
  AT_ASSERT(stream_index <= syclStreamsPerPool);
  return SYCLStreamImplToSYCLStream(reserve_streams[device_index][stream_index-1].get());
}

std::ostream& operator<<(std::ostream& stream, const SYCLStream& s) {
    return stream << s.unwrap();
}

} // namespace sycl
} // namespace at
