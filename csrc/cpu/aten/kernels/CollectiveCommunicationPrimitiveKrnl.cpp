#ifdef USE_CCL
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <aten/CollectiveCommunicationPrimitive.h>
#include <comm/messager.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

namespace {
at::Tensor all_reduce_add_kernel_impl(at::Tensor& t_in) {
  Messenger::getInstance().reduceAdd(t_in);
  return t_in;
}

at::Tensor allgather_kernel_impl(
    at::Tensor t_in,
    std::vector<int64_t> cols_per_rank,
    int64_t world_size) {
  std::vector<at::Tensor> output_tensors;
  auto shape = t_in.contiguous().sizes();
  for (int64_t rank = 0; rank < world_size; rank++) {
    std::vector<int64_t> t_out_shape(shape.begin(), shape.end() - 1);
    t_out_shape.push_back(cols_per_rank[rank + 1] - cols_per_rank[rank]);
    output_tensors.push_back(at::empty(t_out_shape, t_in.options()));
  }

  return Messenger::getInstance().allgather(t_in, output_tensors);
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(all_reduce_add_kernel_stub, &all_reduce_add_kernel_impl);

IPEX_REGISTER_DISPATCH(allgather_kernel_stub, &allgather_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
#endif