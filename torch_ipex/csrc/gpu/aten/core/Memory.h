#pragma once

#include <CL/sycl.hpp>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <core/Stream.h>

DPCPP_DEF_K1(memory_copy);
#define SyclConvertToActualTypePtr(Scalar, buf_acc) \
  static_cast<Scalar*>(static_cast<void*>(((buf_acc.get_pointer().get()))))
#define SyclConvertToActualOffset(Scalar, offset) offset / sizeof(Scalar)

namespace at {
namespace dpcpp {

using buffer_data_type_t = uint8_t;

enum dpcppMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind);
void dpcppMemcpyAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind);

void dpcppMemset(void* data, int value, size_t n_bytes);
void dpcppMemsetAsync(void* data, int value, size_t n_bytes);
void* dpcppMalloc(size_t n_bytes);
void dpcppFree(void* ptr);
void dpcppFreeAll();

/*
 * When we need to write a dpcpp kernel, we need to create DPCPPAccessor object
 * first, then
 * use get_pointer() to get the raw pointer in the dpcpp kernel code. If we need
 * to use
 * cgh handler like .copy(), .fill() which only accept accessor, we can use
 * .get_access()
 * to get accessor.
 * Example:
 *   Acc = DPCPPAccessor(cgh, virtual_ptr, n_bytes);
 *   kernel{
 *     acc_ptr =  Acc.template get_pointer<data_type>();
 *     acc_ptr[i] = ...
 *   }
 */

template <DPCPP::access::mode AccMode, typename ScalarT = uint8_t>
class DPCPPAccessor {
  static constexpr auto global_access = DPCPP::access::target::global_buffer;
  using Accessor = DPCPP::accessor<ScalarT, 1, AccMode, global_access>;

 public:
  DPCPPAccessor(DPCPP::handler& cgh, DPCPP::buffer<ScalarT, 1>& buffer)
      : accessor_(buffer.template get_access<AccMode, global_access>(cgh)) {}

  DPCPPAccessor(DPCPP::handler& cgh, const void* virtual_ptr)
      : offset_(dpcppGetBufferMap().get_offset(virtual_ptr)),
        accessor_(dpcppGetBufferMap()
                      .template get_buffer<ScalarT>(virtual_ptr)
                      .template get_access<AccMode, global_access>(cgh)) {}

  DPCPPAccessor(DPCPP::handler& cgh, const void* virtual_ptr, size_t n_bytes)
      : offset_(0),
        accessor_(dpcppGetBufferMap()
                      .template get_buffer<ScalarT>(virtual_ptr)
                      .template get_access<AccMode, global_access>(
                          cgh,
                          DPCPP::range<1>(n_bytes),
                          DPCPP::id<1>(dpcppGetBufferMap().get_offset(virtual_ptr)))) {}

  const Accessor& get_access() const {
    return accessor_;
  }

  // get_pointer should only be used in dpcpp kernel.
  template <typename T>
  typename DPCPP::global_ptr<T>::pointer_t const get_pointer() const {
    return (typename DPCPP::global_ptr<T>::pointer_t const)(
        SyclConvertToActualTypePtr(T, accessor_) +
        SyclConvertToActualOffset(T, offset_));
  }

 private:
  ptrdiff_t offset_;
  Accessor accessor_;
};

template <typename out_data_type, typename in_data_type>
cl::sycl::buffer<out_data_type, 1> make_buffer(in_data_type virtual_ptr) {
  //get the uint8_t sycl buffer from the vptr.
  auto raw_buf = at::dpcpp::dpcppGetBufferMap().
      template get_buffer<uint8_t>((void*)virtual_ptr);
  auto offset = at::dpcpp::dpcppGetBufferMap().get_offset(virtual_ptr);
  assert(offset >= 0 && "the sycl buffer offset must >= 0");
  auto range = cl::sycl::range<1>(raw_buf.get_size() - offset);
  auto buf = cl::sycl::buffer<uint8_t, 1>(raw_buf,
                                          cl::sycl::id<1>{static_cast<size_t>(offset)},
                                          range);

  //reinterpret the buffer to the required type.
  range = cl::sycl::range<1>(buf.get_size()/sizeof(out_data_type));
  return buf.template reinterpret<out_data_type, 1>(range);
}

#ifdef USE_USM
template <DPCPP::access::mode acc_mode, typename in_data_type, typename out_data_type = in_data_type>
out_data_type* get_buffer(DPCPP::handler& cgh, in_data_type* virtual_ptr) {
  return static_cast<out_data_type*>(virtual_ptr);
}
#else
template <DPCPP::access::mode acc_mode, typename in_data_type, typename out_data_type = in_data_type>
DPCPP::accessor<out_data_type, 1, acc_mode, DPCPP::access::target::global_buffer>
get_buffer(DPCPP::handler& cgh, in_data_type* virtual_ptr) {
  auto sycl_buf = make_buffer<out_data_type>(virtual_ptr);
  return sycl_buf.template get_access<acc_mode, DPCPP::access::target::global_buffer>(cgh);
}
#endif

template <typename buffer_data_type>
static inline buffer_data_type* get_pointer(buffer_data_type* virtual_ptr) {
  return virtual_ptr;
}

template <typename buffer_data_type, DPCPP::access::mode acc_mode>
static inline buffer_data_type*
get_pointer(DPCPP::accessor<buffer_data_type, 1, acc_mode, DPCPP::access::target::global_buffer> accessor) {
  return accessor.template get_pointer();
}

template <typename scalar1, typename scalar2>
static inline void
dpcppMemoryCopyType(scalar1* dst, const scalar2* src, size_t n_elements) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads =
    dpcpp_queue.get_device().template get_info<dpcpp_dev_max_wgroup_size>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = get_buffer<dpcpp_discard_w_mode>(cgh, src);
    auto out_data = get_buffer<dpcpp_r_mode>(cgh, dst);
    cgh.parallel_for<DPCPP_K(memory_copy, scalar1, scalar2)>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
        auto in_ptr = get_pointer(in_data);
        auto out_ptr = get_pointer(out_data);
        auto id = itemId.get_id(0);
        for (auto i = id; i < n_elements; i += itemId.get_range()[0])
          out_ptr[i] = (scalar1)in_ptr[i];
      });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace dpcpp
} // namespace at
