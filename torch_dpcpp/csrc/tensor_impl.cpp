#include "torch_dpcpp/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace torch_ipex {
namespace {

thread_local c10::Device g_current_device(at::DeviceType::DPCPP, 0);

struct IPEXGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::DPCPP; }

  c10::Device exchangeDevice(c10::Device device) const override {
    std::swap(g_current_device, device);
    return device;
  }

  c10::Device getDevice() const override { return g_current_device; }

  void setDevice(c10::Device device) const override {
    g_current_device = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    g_current_device = device;
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_current_device);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
};

C10_REGISTER_GUARD_IMPL(DPCPP, IPEXGuardImpl);

}  // namespace

IPEXTensorImpl::IPEXTensorImpl(const at::Tensor& tensor) :
    c10::TensorImpl(c10::TensorTypeSet(c10::TensorTypeId::DPCPPTensorId),
                    tensor.dtype(),
                    c10::Device(c10::DeviceType::DPCPP, 0)) {}

}  // namespace torch_ipex
