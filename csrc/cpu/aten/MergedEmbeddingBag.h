#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

namespace {
using namespace at;
enum PoolingMode { SUM = 0, MEAN = 1 };
template <class T>
class EMBROW {
 public:
  T* data = nullptr;
  int32_t length;
  EMBROW(int32_t len) {
    length = len;
    Allocator* allocator = c10::GetAllocator(c10::DeviceType::CPU);
    data = (T*)allocator->raw_allocate(len * sizeof(T));
    memset(data, 0, len * sizeof(T));
  }
  ~EMBROW() {
    Allocator* allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(data);
  }
};

template <class T>
class EmbeddingGradCache {
 public:
  std::unordered_map<int32_t, EMBROW<T>> cache;
};

struct SGDArgs {
  SGDArgs(const TensorList& bf16_trail_, float weight_decay_, float lr_)
      : bf16_trail(bf16_trail_), weight_decay(weight_decay_), lr(lr_) {}

  TensorList bf16_trail;
  float weight_decay;
  float lr;
};

template <typename data_t, typename acc_t, typename optimizer_args_t>
class EmbeddingGradUpdate {};

template <typename data_t, typename acc_t>
class EmbeddingGradUpdate<data_t, acc_t, SGDArgs> {
 public:
  static void update(
      data_t* weight,
      const EmbeddingGradCache<acc_t>& egc,
      const SGDArgs& args,
      const int32_t table_id,
      const int64_t emb_dim);
};

std::vector<Tensor> merged_embeddingbag_forward_cpu_kernel_impl(
    const std::vector<Tensor>& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets);

std::vector<Tensor> merged_embeddingbag_backward_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets);

void merged_embeddingbag_backward_sgd_cpu_kernel_impl(
    const TensorList& grad_outs_,
    TensorList weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& bf16_trail,
    const double weight_decay,
    const double lr);

} // namespace

using merged_embeddingbag_forward_cpu_kernel_fn = std::vector<Tensor> (*)(
    const std::vector<Tensor>&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool);
DECLARE_DISPATCH(
    merged_embeddingbag_forward_cpu_kernel_fn,
    merged_embeddingbag_forward_cpu_kernel_stub);

using merged_embeddingbag_backward_cpu_kernel_fn = std::vector<Tensor> (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool);
DECLARE_DISPATCH(
    merged_embeddingbag_backward_cpu_kernel_fn,
    merged_embeddingbag_backward_cpu_kernel_stub);

using merged_embeddingbag_backward_sgd_cpu_kernel_fn = void (*)(
    const TensorList&,
    TensorList,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool,
    const TensorList&,
    const double,
    const double);
DECLARE_DISPATCH(
    merged_embeddingbag_backward_sgd_cpu_kernel_fn,
    merged_embeddingbag_backward_sgd_cpu_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
