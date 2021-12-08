#include "Softmax.h"
#include "csrc/cpu/ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

// softmax kernel for inference mode with oneDNN implementation
at::Tensor softmax_impl(
    const at::Tensor& input,
    const int64_t dim,
    const bool inplace) {
  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  auto input_ = input.is_contiguous() ? input : input.contiguous();
  ideep::tensor mkldnn_input = itensor_view_from_dense(input_);
  auto output = inplace ? input_ : at::empty_like(input_);
  ideep::tensor mkldnn_output = itensor_view_from_dense(output);
  ideep::softmax_forward::compute(mkldnn_input, mkldnn_output, wrapped_dim);
  return output;
}

} // namespace cpu
} // namespace torch_ipex
