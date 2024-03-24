#include <core/Convertor.h>
#include <core/PreInitHook.h>
#include <runtime/Device.h>

namespace torch_ipex::xpu {
namespace dpcpp {
namespace detail {

// Parameter `device_id` is a device index in sycl::device::get_devices(). It is
// unused but a reserved parameter.
at::Device getATenDeviceFromUSM(void* src, const DeviceIndex device_id) {
  auto& default_ctx = at::xpu::get_device_context();
  // Check that pointer is known in the context
  sycl::usm::alloc alloc_type = sycl::get_pointer_type(src, default_ctx);

  TORCH_CHECK(
      alloc_type != sycl::usm::alloc::unknown,
      "Data pointer is not bound to the default context of the specific device.");
  TORCH_CHECK(
      alloc_type == sycl::usm::alloc::device,
      "XPU only support sycl::usm::alloc::device type pointer.");

  auto raw_device_id = at::xpu::get_device_idx_from_pointer(src);
  return at::Device(c10::DeviceType::XPU, raw_device_id);
}
} // namespace detail

// We should minimize the number of calling dpcpp runtime APIs as much as
// possible, because dpcpp's runtime overhead may neutralize the benefits of
// zero copy when the data need to convert is little.
// NOTE: If the root device can be partitioned into two sub-devices, like PVC,
// in `COMPOSITE` mode. The users have to use disable_tile_as_device() to
// disable the partition feature when the src pointer is not allocated in the
// sub-device. Otherwise, the users have to allocate the src pointer with a
// specified tile supplied by the IPEX device. Parameter device_id is a root
// device index in sycl::get_devices. It is unused but a reserved parameter.
// Parameter `device_id` is a device index in sycl::device::get_devices(). It is
// unused but a reserved parameter.
Tensor fromUSM(
    void* src,
    const ScalarType stype,
    IntArrayRef shape,
    c10::optional<IntArrayRef> strides,
    const DeviceIndex device_id) {
  at::Device device = detail::getATenDeviceFromUSM(src, device_id);

  if (!strides) {
    return at::from_blob(
        src, shape, nullptr, at::device(device).dtype(stype), {device});
  }

  return at::from_blob(
      src,
      shape,
      strides.value_or(IntArrayRef{}),
      nullptr,
      at::device(device).dtype(stype),
      {device});
}

void* toUSM(const Tensor& src) {
  TORCH_CHECK(src.is_xpu(), "src must be a XPU tensor.");
  return src.data_ptr();
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
