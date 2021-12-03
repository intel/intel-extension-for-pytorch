#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <torch/extension.h>
#include "cpu/bf16/vec/bf16_vec_kernel.h"
#include "cpu/utils/csr2csc.h"

namespace torch_ipex {
namespace cpu {

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

} // namespace cpu
} // namespace torch_ipex
