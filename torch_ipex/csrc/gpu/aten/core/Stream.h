#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#include <core/DPCPPUtils.h>
#include <core/Macros.h>

namespace at {
namespace dpcpp {

#define dpcppStream_t unsigned long

#define DPCPP_STREAM_COMPUTATION_INDEX 0
#define DPCPP_STREAM_IO_INDEX 1
#define DPCPP_STREAM_NETWORK_INDEX 2
#define DPCPP_STREAM_MAX_INDEX 32

class AT_DPCPP_API DPCPPStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit DPCPPStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::DPCPP);
  }

  explicit DPCPPStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const DPCPPStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const DPCPPStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  operator Stream() const {
    return unwrap();
  }

  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  Device device() const {
    return Device(DeviceType::DPCPP, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  Stream unwrap() const {
    return stream_;
  }

  uint64_t pack() const noexcept {
    return stream_.pack();
  }

  static DPCPPStream unpack(uint64_t bits) {
    return DPCPPStream(Stream::unpack(bits));
  }

  DPCPP::queue& dpcpp_queue() const;

  dpcppStream_t stream() const {
    return (dpcppStream_t)this->dpcpp_queue().get();
  }

 private:
  Stream stream_;
};

CAFFE2_API DPCPPStream getDPCPPStreamFromPool(
    const bool isDefault = false,
    DeviceIndex device_index = -1);

CAFFE2_API DPCPPStream getDefaultDPCPPStream(DeviceIndex device_index = -1);

CAFFE2_API DPCPPStream getCurrentDPCPPStream(DeviceIndex device_index = -1);

CAFFE2_API void setCurrentDPCPPStream(DPCPPStream stream);

CAFFE2_API DPCPPStream
getDPCPPStreamOnDevice(DeviceIndex device_index, int stream_index);

C10_API std::ostream& operator<<(std::ostream& stream, const DPCPPStream& s);

} // namespace dpcpp
} // namespace at

namespace std {
template <>
struct hash<at::dpcpp::DPCPPStream> {
  size_t operator()(at::dpcpp::DPCPPStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
