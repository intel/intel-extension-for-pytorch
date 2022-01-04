#include "Softmax.h"
#include "csrc/cpu/ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

// softmax kernel for inference mode with oneDNN implementation
at::Tensor softmax_impl(const at::Tensor& input, const int64_t dim) {
  // do not support non-contiguous input, which should go into aten::softmax
  TORCH_CHECK(
      input.is_contiguous(),
      "ipex::softmax: Expected contiguous tensor input!");
  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  auto output = at::empty_like(input);
  ideep::tensor mkldnn_output = itensor_view_from_dense(output);
  ideep::softmax_forward::compute(mkldnn_input, mkldnn_output, wrapped_dim);
  return output;
}

// inplace softmax kernel for inference mode with oneDNN implementation
void softmax_impl_(at::Tensor& input, const int64_t dim) {
  // do not support non-contiguous input, which should go into aten::softmax
  TORCH_CHECK(
      input.is_contiguous(),
      "ipex::softmax_: Expected contiguous tensor input!");
  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  ideep::tensor mkldnn_output = itensor_view_from_dense(input);
  ideep::softmax_forward::compute(mkldnn_input, mkldnn_output, wrapped_dim);
}

} // namespace cpu
} // namespace torch_ipex
