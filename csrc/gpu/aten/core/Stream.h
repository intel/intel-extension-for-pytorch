#pragma once

#include <c10/core/Stream.h>

#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

class DPCPPStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit DPCPPStream(Stream stream);

  // Construct a DPCPPStream from a Stream with no error checking.
  explicit DPCPPStream(Unchecked, Stream stream);

  bool operator==(const DPCPPStream& other) const noexcept;

  bool operator!=(const DPCPPStream& other) const noexcept;

  operator Stream() const;

  DeviceIndex device_index() const;

  Device device() const;

  StreamId id() const;

  void synchronize() const;

  void synchronize_and_throw() const;

  Stream unwrap() const;

  uint64_t pack() const noexcept;

  static DPCPPStream unpack(uint64_t bits) {
    return DPCPPStream(Stream::unpack(bits));
  }

  void* opaque() const;

 private:
  Stream stream_;
};

DPCPPStream getDPCPPStreamFromPool(bool is_default, DeviceIndex device_index);

DPCPPStream getDefaultDPCPPStream(DeviceIndex device_index = -1);

DPCPPStream getCurrentDPCPPStream(DeviceIndex device_index = -1);

void setCurrentDPCPPStream(DPCPPStream stream);

} // namespace dpcpp
} // namespace xpu

using namespace xpu::dpcpp;

namespace std {
template <>
struct hash<DPCPPStream> {
  size_t operator()(DPCPPStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
