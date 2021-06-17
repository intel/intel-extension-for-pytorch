#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/core/DeviceGuard.h>

#include <utils/DPCPP.h>
#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

class IPEX_API DPCPPStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit DPCPPStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::XPU);
  }

  // Construct a DPCPPStream from a Stream with no error checking.
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
    return Device(DeviceType::XPU, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    dpcpp_queue().wait();
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

 private:
  Stream stream_;
};

DPCPPStream getDPCPPStreamFromPool(bool is_default, DeviceIndex device_index);

DPCPPStream getDefaultDPCPPStream(DeviceIndex device_index = -1);

IPEX_API DPCPPStream getCurrentDPCPPStream(DeviceIndex device_index = -1);

IPEX_API void setCurrentDPCPPStream(DPCPPStream stream);

} // namespace dpcpp
} // namespace xpu

namespace std {
template <>
struct hash<xpu::dpcpp::DPCPPStream> {
  size_t operator()(xpu::dpcpp::DPCPPStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
