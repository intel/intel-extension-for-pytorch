#pragma once

#include <runtime/DPCPPUtils.h>
#include <core/Stream.h>


DPCPP_DEF_K1(memory_copy);
#define SyclConvertToActualTypePtr(Scalar, buf_acc) \
  static_cast<Scalar*>(static_cast<void*>(((buf_acc.get_pointer().get()))))
#define SyclConvertToActualOffset(Scalar, offset) offset / sizeof(Scalar)

namespace xpu {
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

template <DPCPP::access::mode acc_mode, typename in_data_type, typename out_data_type = in_data_type>
out_data_type* get_buffer(DPCPP::handler& cgh, in_data_type* virtual_ptr) {
  return static_cast<out_data_type*>(virtual_ptr);
}

template <typename buffer_data_type>
static inline buffer_data_type* get_pointer(buffer_data_type* ptr) {
  return ptr;
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
