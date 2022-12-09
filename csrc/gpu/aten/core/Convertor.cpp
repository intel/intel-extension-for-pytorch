#include <core/Convertor.h>
#include <core/PreInitHook.h>
#include <runtime/Device.h>

namespace xpu {
namespace dpcpp {
namespace detail {

// Parameter device_id is index of the root device in sycl::get_devices. It is
// unused but a reserved parameter.
at::Device getATenDeviceFromUSM(void* src, const DeviceIndex device_id) {
#if defined(USE_MULTI_CONTEXT)
  // We can't get the context created by external libaray. So we can't find
  // which tile the src is allocated in.
  TORCH_INTERNAL_ASSERT(false);
#endif
  auto default_ctx = xpu::dpcpp::dpcppGetDeviceContext();
  // Check that pointer is known in the context
  sycl::usm::alloc alloc_type = sycl::get_pointer_type(src, default_ctx);

  TORCH_CHECK(
      alloc_type != sycl::usm::alloc::unknown,
      "Data pointer is not bound to the default context of the specific device.");
  TORCH_CHECK(
      alloc_type == sycl::usm::alloc::device,
      "XPU only support sycl::usm::alloc::device type pointer.");

  // Get sycl::device where the data was allocated, but if src is allocated with
  // a default device and without a specified tile, raw_device will be a parent
  // device, which can't be enumerated in IPEX.
  auto raw_device = sycl::get_pointer_device(src, default_ctx);
  // It is possible that _lazy_init has not been called. Then we need to make
  // sure we have called _lazy_init here.
  do_pre_init_hook();
  auto raw_device_id = xpu::dpcpp::dpcppGetDeviceIndex(raw_device);
  // When src is allocated without a specified tile, we will can't find which
  // tile is src in.
  TORCH_CHECK(
      raw_device_id != -1,
      "The IPEX cannot find the sycl device becuase it is not enumerated in IPEX.");
  return at::Device(c10::DeviceType::XPU, raw_device_id);
}
} // namespace detail

// We should minimize the number of calling dpcpp runtime APIs as much as
// possible, because dpcpp's runtime overhead may neutralize the benefits of
// zero copy when the data need to convert is little.
// NOTE: If the root device can be partitioned into two sub-devices, like PVC.
// The users have to use disable_tile_as_device() to disable the partition
// feature when the src pointer is not allocated in the sub-device. Otherwise,
// the users have to allocate the src pointer with a specified tile supplied by
// the IPEX device.
// Parameter device_id is a root device index in sycl::get_devices. It is unused
// but a reserved parameter.
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
} // namespace xpu
