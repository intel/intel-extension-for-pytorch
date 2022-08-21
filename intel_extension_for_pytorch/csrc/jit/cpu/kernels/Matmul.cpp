#include "Matmul.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/utils/ipex_op_profile.h"
#include "mkl.h"

namespace torch_ipex {
namespace cpu {

/**
 * MKL FP32 BMM kernel
 *
 * Restrictions on the Transpose-free MKL BMM kernel:
 * 1. Minimum stride of the input and output tensors should be 1.
 * 2. The input tensors should have the minimum stride (1) at the
 *    last index (-1) or the second last index (-2).
 * 3. The last stride of the output tensor should be 1.
 * 4. The first stride of the input and output tensors should be
 *    the largest, which means the first dimension should be the highest.
 * 5. The dimension number of the input and output tensors should be the
 *    same and larger than 2.
 * If any tensor does not meet one of the above the conditions, make sure
 * to apply contiguous() before sending it into this BMM kernel.
 **/
void mkl_fp32_bmm_impl(
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Tensor& out,
    const double& output_scale) {
#define GRP_COUNT 1

  auto batch_dim = batch1.dim();

  MKL_INT m[GRP_COUNT] = {batch1.size(-2)};
  MKL_INT k[GRP_COUNT] = {batch1.size(-1)};
  MKL_INT n[GRP_COUNT] = {batch2.size(-1)};

  MKL_INT lda[GRP_COUNT] = {
      batch1.stride(-1) == 1 ? batch1.stride(-2) : batch1.stride(-1)};
  MKL_INT ldb[GRP_COUNT] = {
      batch2.stride(-1) == 1 ? batch2.stride(-2) : batch2.stride(-1)};
  MKL_INT ldc[GRP_COUNT] = {out.stride(-2)};

  CBLAS_TRANSPOSE transA[GRP_COUNT] = {
      batch1.stride(-1) == 1 ? CblasNoTrans : CblasTrans};
  CBLAS_TRANSPOSE transB[GRP_COUNT] = {
      batch2.stride(-1) == 1 ? CblasNoTrans : CblasTrans};

  float alpha[GRP_COUNT] = {output_scale};
  float beta[GRP_COUNT] = {0.0};

  int64_t array_size = batch1.numel() / (batch1.size(-2) * batch1.size(-1));
  const MKL_INT size_per_grp[GRP_COUNT] = {array_size};
  float *a_array[array_size], *b_array[array_size], *c_array[array_size];

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int64_t i = 0; i < array_size; ++i) {
    a_array[i] = batch1.data_ptr<float>();
    b_array[i] = batch2.data_ptr<float>();
    c_array[i] = out.data_ptr<float>();
    int64_t count = 1;
    for (int64_t j = batch_dim - 3; j >= 0; --j) {
      a_array[i] += ((int64_t)(i / count) % batch1.size(j)) * batch1.stride(j);
      b_array[i] += ((int64_t)(i / count) % batch2.size(j)) * batch2.stride(j);
      c_array[i] += ((int64_t)(i / count) % out.size(j)) * out.stride(j);
      count *= batch1.size(j);
    }
  }

  cblas_sgemm_batch(
      CblasRowMajor,
      transA,
      transB,
      m,
      n,
      k,
      alpha,
      (const float**)a_array,
      lda,
      (const float**)b_array,
      ldb,
      beta,
      c_array,
      ldc,
      GRP_COUNT,
      size_per_grp);
}

/**
 * bmm oneDNN kernel
 *
 * @param tensor1
 * @param tensor2
 * @param out Optinal output provided by user for matmul
 * @attr Attribute for matmul oneDNN primitive
 * @return output Tensor.
 * Since oneDNN 2.6.0, AMX and AVX512 brgemm are enabled for the DNNL MATMUL
 * primitive if the input tensors are with the following tags:
 * 3-dim - abc, acb; 4-dim - abcd, acbd, adbc, abdc.
 * If the input tensor has one of the above layouts, the contiguous should NOT
 * be applied to avoid unnecessary transpose (copy).
 * The MKL BMM kernel has better FP32 performance than that of the DNNL MATMUL
 * primitive, and it allows the input tensors can be a part of the other bigger
 * tensor. Thus the QKV split is NOT required for the FP32 MHA matmul.
 *
 * Since the MKL BMM kernel cannot fuse any post-OP, for the cases 1. FP32 BMM
 * with any DNNL-defined post-OP, 2. BF16 BMM, the DNNL MATMUL primitive is
 * applied. For FP32 BMM with mul/div, the MKL BMM kernel is applied.
 **/
at::Tensor bmm_impl(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& postop_tensors,
    const float dst_coeff = 1.0f) {
  // The following conditions are strict to exclude some extreme cases when the
  // tensors have the undefined stride values. For the sake of reliability of
  // transpose-free Matmul kernel, contiguous will be applied to these tensors.
  auto check_tensor_dim_stride = [](at::Tensor tensor) {
    // Check if the Tensor is 3-dim or 4-dim
    if (tensor.dim() != 3 && tensor.dim() != 4)
      return false;
    // Check the strides of the tensor are not out of the tensor's ranges.
    if (tensor.stride(0) * tensor.size(0) != tensor.numel())
      return false;
    return true;
  };
  auto check_tensor_layout = [](at::Tensor tensor) {
    // Check if 'a' is the first dim
    for (int64_t i = 1; i < tensor.dim(); ++i) {
      if (tensor.stride(0) < tensor.stride(i))
        return false;
    }
    // Check the minimum stride is at the last or second last index
    // and its value is 1.
    if (!(tensor.stride(-1) == 1 || tensor.stride(-2) == 1))
      return false;
    return true;
  };

  auto output = out;
  if (!out.defined()) {
    const int64_t dim = tensor1.dim();
    std::vector<int64_t> output_size(dim);
    for (auto i = 0; i < dim - 1; i++) {
      output_size[i] = tensor1.size(i);
    }
    output_size[dim - 1] = tensor2.size(dim - 1);
    output = at::empty(output_size, tensor1.options());
  }

  if (tensor1.dtype() == at::kBFloat16 || attr.has_post_op()) {
    auto tensor1_ =
        (check_tensor_dim_stride(tensor1) && check_tensor_layout(tensor1))
        ? tensor1
        : tensor1.contiguous();
    auto tensor2_ =
        (check_tensor_dim_stride(tensor2) && check_tensor_layout(tensor2))
        ? tensor2
        : tensor2.contiguous();

    const ideep::tensor mkldnn_input = itensor_view_from_dense(tensor1_);
    const ideep::tensor mkldnn_tensor2 = itensor_view_from_dense(tensor2_);
    ideep::tensor mkldnn_output = itensor_view_from_dense(output);

    ideep::matmul_forward::compute(
        mkldnn_input,
        mkldnn_tensor2,
        mkldnn_output,
        dst_coeff,
        1.0,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr,
        postop_tensors);
  } else {
    auto tensor1_ =
        check_tensor_layout(tensor1) ? tensor1 : tensor1.contiguous();
    auto tensor2_ =
        check_tensor_layout(tensor2) ? tensor2 : tensor2.contiguous();

    mkl_fp32_bmm_impl(tensor1_, tensor2_, output, dst_coeff);
  }

  return output;
}

at::Tensor dil_matmul(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  IPEX_RECORD_FUNCTION("dil_matmul", c10::ArrayRef<c10::IValue>({}));

  return bmm_impl(tensor1, tensor2, at::Tensor(), ideep::attr_t(), {}, 1.f);
}

/**
 * Dispatch at::matmul + at::div pattern to ipex for jit inference, but only
 * one-element tensor and channel dim boadcast is enabled in oneDNN 2.2.0 now.
 * So, for simplicity,this path is just a fallback path now. output(out) =
 * (tensor1 * tensor2).div(div_input)
 *
 * @param tensor1
 * @param tensor2
 * @param out Optinal output provided by user for matmul
 * @param div_input Input Tensor for div
 * @return Value for the fusion pattern output.
 */
at::Tensor dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const at::Tensor& div_input) {
  IPEX_RECORD_FUNCTION(
      "dil_matmul_div_fallback", c10::ArrayRef<c10::IValue>({}));

  if (out.defined()) {
    at::matmul_out(out, tensor1, tensor2);
    return out.div_(div_input);
  }
  auto output = at::matmul(tensor1, tensor2);
  return output.div_(div_input);
}

/**
 *Dispatch at::matmul + at::div pattern to ipex for jit inference, but only bmm
 *with same shape for tensor1 and tensor2 and scalar input for div will be
 *dispatched to oneDNN kernel. Otherwise will fallback. For oneDNN kernel,
 *scalar input will be used as the scale attribute for matmul primitive.
 *output(out) = (tensor1 * tensor2).div(div_input_scalar).
 *ToDo: matmul + div scalar for matmul with other shape
 *
 *@param tensor1
 *@param tensor2
 *@param out Optinal output provided by user for matmul
 *@param div_input Input scalar for div
 *@return Value for the fusion pattern output.
 */
at::Tensor dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const c10::Scalar& div_input) {
  IPEX_RECORD_FUNCTION("dil_matmul_div_scalar", c10::ArrayRef<c10::IValue>({}));

  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  if (dim_tensor1 == dim_tensor2 && dim_tensor1 >= 3) {
    float scale = 1.0f / div_input.to<float>();
    return bmm_impl(tensor1, tensor2, out, ideep::attr_t(), {}, scale);
  } else {
    return dil_matmul_div(
        tensor1, tensor2, out, at::native::wrapped_scalar_tensor(div_input));
  }
}

at::Tensor dil_bmm_add(
    const at::Tensor& input,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const c10::Scalar& alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_bmm_add", c10::ArrayRef<c10::IValue>({}));
#endif
  auto batch1_dim = batch1.dim();
  auto batch2_dim = batch2.dim();
  if (batch1_dim == batch2_dim && batch1_dim >= 3) {
    auto _input = input.is_contiguous() ? input : input.contiguous();
    ideep::tensor onednn_input = itensor_view_from_dense(_input);

    auto op_attr = ideep::attr_t::fuse_binary(
        dnnl::algorithm::binary_add, onednn_input.get_desc());
    return bmm_impl(
        batch1, batch2, at::Tensor(), op_attr, {onednn_input}, 1.0f);
  } else {
    return at::baddbmm(input, batch1, batch2);
  }
}

} // namespace cpu
} // namespace torch_ipex
