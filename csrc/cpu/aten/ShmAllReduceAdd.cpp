#ifdef USE_CCL
#include "ShmAllReduceAdd.h"
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(shm_all_reduce_add_kernel_stub);

at::Tensor shm_all_reduce_add_forward_cpu(
    at::Tensor& t_in,
    at::Tensor& t_address,
    at::Tensor& t_state,
    at::Tensor& t_blockState,
    int64_t shm_block_size,
    int64_t rank,
    int64_t world_size) {
  return shm_all_reduce_add_kernel_stub(
      kCPU,
      t_in,
      t_address,
      t_state,
      t_blockState,
      shm_block_size,
      rank,
      world_size);
}

} // namespace cpu
} // namespace torch_ipex
#endif