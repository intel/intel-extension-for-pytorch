#pragma once

#include <utils/DPCPP.h>
#include <core/Stream.h>

DPCPP_DEF_K1(memory_copy);

namespace xpu {
namespace dpcpp {

enum dpcppMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

void dpcppMemcpy(void* dst, const void* src, size_t n_bytes, dpcppMemcpyKind kind);

void dpcppMemcpyAsync(void* dst, const void* src, size_t n_bytes, dpcppMemcpyKind kind);

void dpcppMemset(void* data, int value, size_t n_bytes);

void dpcppMemsetAsync(void* data, int value, size_t n_bytes);

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

} // namespace dpcpp
} // namespace at
