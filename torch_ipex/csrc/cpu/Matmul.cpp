#include "Linear.h"
#include "mkldnn/MKLDNNCommon.h"
#include "WeightPrepack.h"

namespace torch_ipex {
namespace cpu {

/**
 * bmm oneDNN kernel 
 * 
 * @param tensor1
 * @param tensor2
 * @param out Optinal output provided by user for matmul
 * @attr Attribute for matmul oneDNN primitive 
 * @return output Tensor.
 */
at::Tensor bmm_impl(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const ideep::attr_t& attr,
    const float dst_coeff = 1.0f) {

  auto tensor1_ = tensor1.is_contiguous() ? tensor1 : tensor1.contiguous();
  auto tensor2_ = tensor2.is_contiguous() ? tensor2 : tensor2.contiguous();
  const int64_t dim = tensor1.dim();
  const ideep::tensor mkldnn_input = at::native::itensor_view_from_dense(tensor1_);
  const ideep::tensor mkldnn_tensor2 = at::native::itensor_view_from_dense(tensor2_);
  
  auto output = out;
  if (!out.defined()) {
    std::vector<int64_t> output_size(dim);
    for (auto i = 0; i <  dim - 1; i++) {
      output_size[i] = tensor1.size(i);
    }
    output_size[dim -1] = tensor2.size(dim - 1);
    output = at::empty(output_size, tensor1.options());
  }  
  ideep::tensor mkldnn_output = at::native::itensor_view_from_dense(output);
  ideep::matmul_forward::compute(
    mkldnn_input,
    mkldnn_tensor2,
    mkldnn_output,
    dst_coeff,
    1.0,
    ideep::scale_t(),
    ideep::scale_t(),
    ideep::scale_t(),
    attr);

  return output;
}

}  // namespace cpu
}  // namespace torch_ipex
