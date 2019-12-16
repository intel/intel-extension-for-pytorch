#include "dpcpp_allocator.h"

#include <c10/util/Exception.h>

#include "cpu/DPCPPCPUAllocator.h"
#include "ipex_tensor_impl.h"

namespace torch_ipex {

DefaultDPCPPAllocator::DefaultDPCPPAllocator(std::shared_ptr<at::Allocator> cpu_alloctor, std::shared_ptr<at::Allocator> gpu_alloctor)
  : m_dpcpp_cpu_allocator(cpu_alloctor), m_dpcpp_gpu_allocator(gpu_alloctor) {}

at::DataPtr DefaultDPCPPAllocator::allocate(size_t nbytes) const {
  return get_current_allocator()->allocate(nbytes);
}

at::DeleterFnPtr DefaultDPCPPAllocator::raw_deleter() const {
  return get_current_allocator()->raw_deleter();
}

std::shared_ptr<at::Allocator> DefaultDPCPPAllocator::get_current_allocator() const {
  TORCH_CHECK(IPEXTensorImpl::GetCurrentAtenDevice().has_index());
  TORCH_CHECK(IPEXTensorImpl::GetCurrentAtenDevice().type() == at::DeviceType::DPCPP);
  if (IPEXTensorImpl::GetCurrentAtenDevice().index() == 0) {
    TORCH_CHECK(m_dpcpp_cpu_allocator != nullptr);
    return m_dpcpp_cpu_allocator;
  } else {
    TORCH_CHECK(m_dpcpp_gpu_allocator != nullptr);
    return m_dpcpp_gpu_allocator;
  }
}

void NoDelete(void*) {}

at::Allocator* GetDPCPPAllocator() {
  return at::GetAllocator(at::DeviceType::DPCPP);
}

void SetDPCPPAllocator(at::Allocator* alloc) {
  SetAllocator(at::DeviceType::DPCPP, alloc);
}

static DefaultDPCPPAllocator g_dpcpp_alloc(std::shared_ptr<at::Allocator>(new cpu::DefaultDPCPPCPUAllocator()), nullptr);

at::Allocator* GetDefaultDPCPPAllocator() {
  return &g_dpcpp_alloc;
}

} // namespace torch_ipex

namespace c10 {

REGISTER_ALLOCATOR(at::DeviceType::DPCPP, &torch_ipex::g_dpcpp_alloc);

} // c10
