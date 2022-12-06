#pragma once

#include <c10/util/Optional.h>
#include <core/Stream.h>
#include <memory>

namespace xpu {
namespace dpcpp {

struct DPCPPEventBase {
  // Constructors
  DPCPPEventBase() {}

  virtual ~DPCPPEventBase(){};

  DPCPPEventBase(const DPCPPEventBase&) = delete;
  DPCPPEventBase& operator=(const DPCPPEventBase&) = delete;

  virtual at::optional<at::Device> device() const = 0;

  virtual DeviceIndex device_index() const = 0;

  virtual bool query() const = 0;

  virtual void record() = 0;

  virtual void record(const DPCPPStream& stream) = 0;

  virtual void recordOnce(const DPCPPStream& stream) = 0;

  virtual void block(const DPCPPStream& stream) = 0;

  virtual void synchronize() = 0;

  virtual float elapsed_time(const DPCPPEventBase* other) = 0;

  virtual void ipc_handle(void* handle) = 0;
};

std::shared_ptr<DPCPPEventBase> create_dpcpp_event();

} // namespace dpcpp
} // namespace xpu
