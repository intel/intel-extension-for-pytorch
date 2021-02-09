#include "dpcpp_allocator.h"

#include <c10/util/Exception.h>

#include "cpu/DPCPPCPUAllocator.h"
#include "ipex_tensor_impl.h"

namespace torch_ipex {

DefaultDPCPPAllocator::DefaultDPCPPAllocator(std::shared_ptr<at::Allocator> cpu_alloctor)
  : m_dpcpp_cpu_allocator(cpu_alloctor) {}

at::DataPtr DefaultDPCPPAllocator::allocate(size_t nbytes) const {
  return get_current_allocator()->allocate(nbytes);
}

at::DeleterFnPtr DefaultDPCPPAllocator::raw_deleter() const {
  return get_current_allocator()->raw_deleter();
}

std::shared_ptr<at::Allocator> DefaultDPCPPAllocator::get_current_allocator() const {
  TORCH_CHECK(IPEXTensorImpl::GetCurrentAtenDevice().has_index());
  TORCH_CHECK(IPEXTensorImpl::GetCurrentAtenDevice().type() == at::DeviceType::XPU);
  TORCH_CHECK(IPEXTensorImpl::GetCurrentAtenDevice().index() == 0);
  TORCH_CHECK(m_dpcpp_cpu_allocator != nullptr);
  return m_dpcpp_cpu_allocator;
}

void NoDelete(void*) {}

at::Allocator* GetDPCPPAllocator() {
  return at::GetAllocator(at::DeviceType::XPU);
}

void SetDPCPPAllocator(at::Allocator* alloc) {
  SetAllocator(at::DeviceType::XPU, alloc);
}

static DefaultDPCPPAllocator g_dpcpp_alloc(std::shared_ptr<at::Allocator>(new cpu::DefaultDPCPPCPUAllocator()));

at::Allocator* GetDefaultDPCPPAllocator() {
  return &g_dpcpp_alloc;
}

} // namespace torch_ipex

namespace c10 {

REGISTER_ALLOCATOR(at::DeviceType::XPU, &torch_ipex::g_dpcpp_alloc);

} // c10
