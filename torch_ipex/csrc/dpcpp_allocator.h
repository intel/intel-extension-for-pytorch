#include <c10/core/Allocator.h>

#include <memory>

namespace torch_ipex {

struct DefaultDPCPPAllocator final : at::Allocator {
  DefaultDPCPPAllocator(std::shared_ptr<at::Allocator>, std::shared_ptr<at::Allocator>);
  ~DefaultDPCPPAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override;
  at::DeleterFnPtr raw_deleter() const override;

 private:
   std::shared_ptr<at::Allocator> get_current_allocator() const;

 protected:
   std::shared_ptr<at::Allocator> m_dpcpp_cpu_allocator;
   std::shared_ptr<at::Allocator> m_dpcpp_gpu_allocator;
};

} // namespace torch_ipex
