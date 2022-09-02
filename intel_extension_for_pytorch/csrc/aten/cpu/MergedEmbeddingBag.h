#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>
#include <torch/all.h>
#include "utils/csr2csc.h"

namespace torch_ipex {
namespace cpu {

namespace {

struct SGDArgs {
  SGDArgs(
      const std::vector<Tensor>& bf16_trail_,
      float weight_decay_,
      float lr_)
      : bf16_trail(bf16_trail_), weight_decay(weight_decay_), lr(lr_) {}

  std::vector<Tensor> bf16_trail;
  float weight_decay;
  float lr;
};

template <typename T, typename optimizer_args_t>
class AccGradUpdate {};

template <typename T>
class AccGradUpdate<T, SGDArgs> {
 public:
  static void update(
      T* weight,
      T* grad,
      const BatchedHyperCompressedSparseColumn& batched_csc,
      int64_t uniq_index_id,
      int64_t weight_offsets,
      int vector_size,
      int table_id,
      const SGDArgs& args);
};

std::vector<Tensor> merged_embeddingbag_forward_cpu_kernel_impl(
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const std::vector<int64_t> pooling_modes);

void merged_embeddingbag_backward_sgd_cpu_kernel_impl(
    const std::vector<Tensor>& grads_y_,
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    std::vector<int64_t> pooling_modes,
    const std::vector<Tensor>& bf16_trail,
    double weight_decay,
    double lr);

} // namespace

using merged_embeddingbag_forward_cpu_kernel_fn = std::vector<Tensor> (*)(
    const Tensor&,
    const Tensor&,
    const std::vector<Tensor>&,
    const std::vector<int64_t>);
DECLARE_DISPATCH(
    merged_embeddingbag_forward_cpu_kernel_fn,
    merged_embeddingbag_forward_cpu_kernel_stub);

using merged_embeddingbag_backward_sgd_cpu_kernel_fn = void (*)(
    const std::vector<Tensor>&,
    const Tensor&,
    const Tensor&,
    const std::vector<Tensor>&,
    const Tensor&,
    const Tensor&,
    std::vector<int64_t>,
    const std::vector<Tensor>&,
    double,
    double);
DECLARE_DISPATCH(
    merged_embeddingbag_backward_sgd_cpu_kernel_fn,
    merged_embeddingbag_backward_sgd_cpu_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
