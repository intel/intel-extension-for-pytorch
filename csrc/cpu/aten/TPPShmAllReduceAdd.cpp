#include "TPPShmAllReduceAdd.h"
#include <torch/all.h>
namespace torch_ipex {
namespace cpu {
IPEX_DEFINE_DISPATCH(tpp_allreduce_kernel_stub);
void tpp_shmallreduce_forward(
    at::Tensor t_in,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
  RECORD_FUNCTION("tpp_all_reduce_add", c10::ArrayRef<c10::IValue>({}));
  return tpp_allreduce_kernel_stub(kCPU, t_in, process_group);
}

} // namespace cpu
} // namespace torch_ipex