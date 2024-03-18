#ifdef USE_CCL
#include "CollectiveCommunicationPrimitive.h"
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(all_reduce_add_kernel_stub);
IPEX_DEFINE_DISPATCH(allgather_kernel_stub);

at::Tensor all_reduce_add(at::Tensor t_in) {
  RECORD_FUNCTION("ipex::all_reduce_add", c10::ArrayRef<c10::IValue>({}));
  return all_reduce_add_kernel_stub(kCPU, t_in);
}

at::Tensor allgather(
    at::Tensor t_in,
    std::vector<int64_t> cols_per_rank,
    int64_t world_size) {
  RECORD_FUNCTION("ipex::allgather", c10::ArrayRef<c10::IValue>({}));
  return allgather_kernel_stub(kCPU, t_in, cols_per_rank, world_size);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("all_reduce_add(Tensor(a!) t_in)-> (Tensor)");
  m.impl(
      "all_reduce_add", c10::DispatchKey::CPU, torch_ipex::cpu::all_reduce_add);
  m.def("allgather(Tensor input, int[] output, int world_size) -> (Tensor)");
  m.impl("allgather", c10::DispatchKey::CPU, torch_ipex::cpu::allgather);
}
} // namespace
#endif