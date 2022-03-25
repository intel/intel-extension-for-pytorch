#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>

#include "UpdateBatch.h"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {
DEFINE_DISPATCH(rnnt_update_batch_kernel_stub);
}
} // namespace torch_ipex

namespace torch_ipex {
namespace kernel {

static bool rnnt_update_batch(
    const at::Tensor& k,
    const at::Tensor& out_lens,
    at::Tensor label_col,
    at::Tensor symbols_added,
    at::Tensor time_idxs,
    at::Tensor blankness_out,
    at::Tensor blankvec_out,
    at::Tensor not_blank_out,
    at::Tensor label_to_put_out,
    at::Tensor label_tensor_out,
    at::Tensor label_for_next_loop_out,
    at::Tensor hidden_0,
    at::Tensor hidden_1,
    const at::Tensor& hidden_prime_0,
    const at::Tensor& hidden_prime_1,
    at::Tensor x,
    at::Tensor f,
    int64_t max_symbols,
    int64_t blank_id,
    int64_t batch_size,
    int64_t _SOS,
    int64_t max_len) {
#if defined(IPEX_DISP_OP)
  printf("IPEX::rnnt_update_batch\n");
#endif
  IPEX_RECORD_FUNCTION("IPEX::rnnt_update_batch", std::vector<c10::IValue>({}));

  /*
  pointer to torch_ipex::cpu::rnnt_update_batch_kernel_impl(
      k,
      out_lens,
      label_col,
      symbols_added,
      time_idxs,
      blankness_out,
      blankvec_out,
      not_blank_out,
      label_to_put_out,
      label_tensor_out,
      label_for_next_loop_out,
      hidden_0,
      hidden_1,
      hidden_prime_0,
      hidden_prime_1,
      x,
      f,
      max_symbols,
      blank_id,
      batch_size,
      _SOS,
      max_len);
  */
  return torch_ipex::cpu::rnnt_update_batch_kernel_stub(
      kCPU,
      k,
      out_lens,
      label_col,
      symbols_added,
      time_idxs,
      blankness_out,
      blankvec_out,
      not_blank_out,
      label_to_put_out,
      label_tensor_out,
      label_for_next_loop_out,
      hidden_0,
      hidden_1,
      hidden_prime_0,
      hidden_prime_1,
      x,
      f,
      max_symbols,
      blank_id,
      batch_size,
      _SOS,
      max_len);
}

} // namespace kernel
} // namespace torch_ipex

namespace {

static auto dispatch = torch::RegisterOperators().op(
    "torch_ipex::rnnt_update_batch",
    &torch_ipex::kernel::rnnt_update_batch);
}
